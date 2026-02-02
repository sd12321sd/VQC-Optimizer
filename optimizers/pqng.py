import pennylane as qml
from pennylane import numpy as np
from typing import Callable, List, Optional, Tuple, Union


class PQNGOptimizer:
    """
    Projective Quantum Natural Gradient (P-QNG) optimizer
    = H-QNG without coefficients (set all a_r = 1).

    Metric:
        G_ij = sum_{r=1}^v  Tr(∂_i ρ P_r) Tr(∂_j ρ P_r)
            = sum_{r=1}^v  ∂_i <P_r> ∂_j <P_r>

        T = G / (2 * sqrt(v))

    Update (with trust-region damping):
        θ' = θ - η (T + λ I)^{-1} ∇L(θ)

    If lu=True:
        - If energy decreases: accept step, λ ↓
        - If energy increases: reject step, λ ↑
    """

    def __init__(
        self,
        qnode: Callable,
        hamiltonian_terms: Union[qml.Hamiltonian, List[Tuple[float, qml.operation.Operator]]],
        eta: float = 0.1,
        lam: float = 1e-6,
        lu: bool = False,
        rcond: float = 1e-10,
        diff_method: str = "parameter-shift",
        device: Optional[qml.devices.Device] = None,
        cache_term_qnodes: bool = True,
    ):
        self.qnode = qnode  # ansatz callable (not necessarily a QNode)
        self.device = device
        self.eta = eta
        self.lam = lam
        self.lu = lu
        self.lam_factor = 10.0
        self.rcond = rcond
        self.diff_method = diff_method
        self.cache_term_qnodes = cache_term_qnodes

        # Parse Hamiltonian terms (we only need the Pauli strings P_r)
        if isinstance(hamiltonian_terms, qml.Hamiltonian):
            self.ops = list(hamiltonian_terms.ops)
        else:
            self.ops = [op for _, op in hamiltonian_terms]

        self.v = len(self.ops)
        if self.v == 0:
            raise ValueError("P-QNG requires at least one Hamiltonian term (Pauli string).")

        self.norm_factor = 2.0 * np.sqrt(float(self.v))

        # cache: term_index -> QNode returning <P_r>
        self._term_qnodes = {}

    # ==========================================================
    # Public API
    # ==========================================================

    def step(self, cost_fn, theta, *args, **kwargs):
        theta = np.asarray(theta, dtype=float)

        dev = self._get_device(cost_fn)

        old_cost = float(cost_fn(theta, *args, **kwargs))
        grad = self._gradient(cost_fn, theta, *args, **kwargs)

        T = self._pqng_metric(theta, dev)
        d = T.shape[0]

        lam = self.lam
        T_reg = T + lam * np.eye(d)
        T_pinv = np.linalg.pinv(T_reg, rcond=self.rcond)

        delta = self.eta * (T_pinv @ grad)
        new_theta = theta - delta
        new_cost = float(cost_fn(new_theta, *args, **kwargs))

        if self.lu:
            if new_cost < old_cost:
                self.lam = max(lam / self.lam_factor, 1e-8)
                return new_theta
            else:
                print(
                    f"[P-QNG] Rejecting step: "
                    f"E_new={new_cost:.6f} >= E_old={old_cost:.6f}. "
                    f"λ → {lam * self.lam_factor:.1e}"
                )
                self.lam = lam * self.lam_factor
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
                "PQNGOptimizer requires a PennyLane device. "
                "Pass device=... to the optimizer, or pass a QNode cost_fn with a .device attribute."
            )
        return dev

    def _gradient(self, cost_fn, theta, *args, **kwargs):
        grad_fn = qml.grad(cost_fn)
        return np.asarray(grad_fn(theta, *args, **kwargs), dtype=float)

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

    def _pqng_metric(self, theta: np.ndarray, dev) -> np.ndarray:
        """
        Projective metric:
            G = sum_r grad(<P_r>) ⊗ grad(<P_r>)
            T = G / (2*sqrt(v))
        """
        n_params = theta.size
        G = np.zeros((n_params, n_params), dtype=float)

        for idx, P_r in enumerate(self.ops):
            expval_qnode = self._get_term_qnode(dev, idx, P_r)
            grad_pr = qml.grad(expval_qnode)(theta)
            G += np.outer(grad_pr, grad_pr)

        return G / self.norm_factor
    
if __name__ == "__main__":
    n_qubits = 4
    dev = qml.device("default.qubit", wires=n_qubits)

    def ansatz(theta):
        idx = 0
        for l in range(2):
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

    opt = PQNGOptimizer(
        qnode=ansatz,
        hamiltonian_terms=H,
        eta=1e-3,
        lam=1e-5,
        lu=True,        
        device=dev
    )

    theta = 0.1 * np.random.randn(2 * n_qubits)
    print(f"Optimizer using device wires: {opt.device.wires}")

    for t in range(10):
        theta = opt.step(energy, theta)
        print(f"iter {t+1:3d} | E = {energy(theta): .6f} | λ = {opt.lam:.2e}")