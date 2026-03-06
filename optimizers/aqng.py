import pennylane as qml
from pennylane import numpy as np
from typing import Callable, List, Optional, Tuple, Union


class AQNGOptimizer:
    """
    A-QNG: convex combination of P-QNG and QNG metrics.

    Let:
        - P-QNG metric (as you defined):   T_P = G_P / (2*sqrt(v))
          where G_P = sum_r (∇⟨P_r⟩)(∇⟨P_r⟩)^T and v = #Hamiltonian terms
        - QNG metric:                      G_Q via qml.metric_tensor on a pure-state circuit

    Define:
        G_A = T_P + mix * (G_Q - T_P) = (1-mix) T_P + mix G_Q

    Update with damping:
        theta' = theta - eta * (G_A + lam I)^(-1) * grad
    """

    def __init__(
        self,
        qnode: Callable,
        hamiltonian_terms: Union[qml.Hamiltonian, List[Tuple[float, qml.operation.Operator]]],
        eta: float = 0.1,
        lam: float = 1e-6,
        mix: float = 0.5,
        lu: bool = False,
        rcond: Optional[float] = 1e-10,
        diff_method: str = "parameter-shift",
        device: Optional[qml.devices.Device] = None,
        approx_qng: Optional[str] = None,      # None | "diag" | "block-diag"
        cache_term_qnodes: bool = True,
    ):
        self.qnode = qnode
        self.device = device
        self.eta = float(eta)
        self.lam = float(lam)
        self.mix = float(mix)
        self.lu = lu
        self.lam_factor = 10.0
        self.rcond = rcond
        self.diff_method = diff_method
        self.approx_qng = approx_qng
        self.cache_term_qnodes = cache_term_qnodes

        if not (0.0 <= self.mix <= 1.0):
            raise ValueError("mix must be in [0,1].")

        # Store only operators P_r (coeffs ignored for P-QNG as per your definition)
        if isinstance(hamiltonian_terms, qml.Hamiltonian):
            self.ops = list(hamiltonian_terms.ops)
        else:
            self.ops = [op for _, op in hamiltonian_terms]

        if len(self.ops) == 0:
            raise ValueError("Hamiltonian must contain at least one term.")

        self.v = int(len(self.ops))  # number of Pauli terms

        # Cache: term index -> expval QNode
        self._term_qnodes = {}

        # Cache: QNG state-qnode + metric function
        self._state_qnode = None
        self._metric_fn = None

    # ==========================================================
    # Public API
    # ==========================================================

    def step(self, cost_fn, theta, *args, **kwargs):
        theta = np.asarray(theta, dtype=float)
        dev_cost = self._get_device(cost_fn)

        old_cost = float(cost_fn(theta, *args, **kwargs))
        grad = self._gradient_flat(cost_fn, theta, *args, **kwargs)  # (p,)

        # --- P-QNG part (EXACTLY match your definition) ---
        Gp_raw = self._pqng_metric_raw(theta, dev_cost)
        Tp = Gp_raw / (2.0 * np.sqrt(float(self.v)))  # T_P = G_P / (2*sqrt(v))

        # --- QNG part from PennyLane metric_tensor (pure-state FS metric) ---
        Gq = self._qng_metric(theta, dev_cost)

        # --- A-QNG mixing: (1-mix) Tp + mix Gq ---
        Ga = (1.0 - self.mix) * Tp + self.mix * Gq

        p = Ga.shape[0]
        Ga_reg = Ga + self.lam * np.eye(p)
        Ga_pinv = np.linalg.pinv(Ga_reg, rcond=self.rcond)

        delta = self.eta * (Ga_pinv @ grad)
        new_theta = theta - delta.reshape(theta.shape)

        new_cost = float(cost_fn(new_theta, *args, **kwargs))

        if self.lu:
            if new_cost < old_cost:
                self.lam = max(self.lam / self.lam_factor, 1e-12)
                return new_theta
            else:
                print(
                    f"[A-QNG] Rejecting step: "
                    f"E_new={new_cost:.6f} >= E_old={old_cost:.6f}. "
                    f"lam → {self.lam * self.lam_factor:.1e}"
                )
                self.lam = self.lam * self.lam_factor
                return theta
        else:
            return new_theta

    # ==========================================================
    # Internals
    # ==========================================================

    def _get_device(self, cost_fn) -> qml.devices.Device:
        if self.device is not None:
            return self.device
        dev = getattr(cost_fn, "device", None)
        if dev is None:
            raise ValueError(
                "AQNGOptimizer requires a PennyLane device. "
                "Pass device=... to the optimizer, or pass a QNode cost_fn with a .device attribute."
            )
        return dev

    def _gradient_flat(self, cost_fn, theta, *args, **kwargs):
        grad_fn = qml.grad(cost_fn)
        g = grad_fn(theta, *args, **kwargs)
        return np.asarray(g, dtype=float).reshape(-1)

    # ---------------- P-QNG raw metric ----------------

    def _get_term_qnode(self, dev, term_index: int, P_r):
        if self.cache_term_qnodes and term_index in self._term_qnodes:
            return self._term_qnodes[term_index]

        @qml.qnode(dev, interface="autograd", diff_method=self.diff_method)
        def expval_qnode(theta_):
            self.qnode(theta_)
            return qml.expval(P_r)

        if self.cache_term_qnodes:
            self._term_qnodes[term_index] = expval_qnode
        return expval_qnode

    def _pqng_metric_raw(self, theta: np.ndarray, dev) -> np.ndarray:
        p = int(np.prod(theta.shape))
        G = np.zeros((p, p), dtype=float)

        for idx, P_r in enumerate(self.ops):
            expval_qnode = self._get_term_qnode(dev, idx, P_r)
            grad_pr = qml.grad(expval_qnode)(theta)
            g_r = np.asarray(grad_pr, dtype=float).reshape(-1)
            G += np.outer(g_r, g_r)

        return G

    # ---------------- QNG metric via qml.metric_tensor (pure state) ----------------

    def _build_state_qnode_and_metric(self, dev_cost):
        """
        Build a pure-state QNode returning qml.state(), then build metric_tensor on it.
        This is the PennyLane-native path for the QNG (Fubini–Study) metric.
        """
        @qml.qnode(dev_cost, interface="autograd", diff_method=self.diff_method)
        def state_qnode(theta_):
            self.qnode(theta_)
            return qml.state()

        metric_fn = qml.metric_tensor(state_qnode, approx=self.approx_qng)
        return state_qnode, metric_fn

    def _qng_metric(self, theta: np.ndarray, dev_cost) -> np.ndarray:
        if self._metric_fn is None:
            self._state_qnode, self._metric_fn = self._build_state_qnode_and_metric(dev_cost)

        Gq = self._metric_fn(theta)
        return np.asarray(Gq, dtype=float)


# ============================================================
# Minimal test
# ============================================================
if __name__ == "__main__":
    n_qubits = 4
    dev = qml.device("default.qubit", wires=n_qubits+1)

    def ansatz(theta):
        idx = 0
        for _ in range(2):
            for w in range(n_qubits):
                qml.RY(theta[idx], wires=w)
                idx += 1
            for w in range(n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])

    terms = []
    J = 1.0
    h = 0.5
    for w in range(n_qubits - 1):
        terms.append((-J, qml.PauliZ(w) @ qml.PauliZ(w + 1)))
    for w in range(n_qubits):
        terms.append((-h, qml.PauliX(w)))
    H = qml.Hamiltonian([c for c, _ in terms], [op for _, op in terms])

    @qml.qnode(dev, interface="autograd")
    def energy(theta):
        ansatz(theta)
        return qml.expval(H)

    opt = AQNGOptimizer(
        qnode=ansatz,
        hamiltonian_terms=H,
        eta=1e-3,
        lam=1e-3,
        mix=1.0,          # QNG metric_tensor
        lu=False,
        rcond=1e-10,
        device=dev,
    )

    theta = 0.1 * np.random.randn(2 * n_qubits)
    print(f"Cost device wires: {dev.wires}")

    for t in range(10):
        theta = opt.step(energy, theta)
        print(f"iter {t+1:3d} | E = {energy(theta): .6f} | lam = {opt.lam:.2e}")