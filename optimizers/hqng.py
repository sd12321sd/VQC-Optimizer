import pennylane as qml
from pennylane import numpy as np
from typing import Callable, List, Optional, Tuple, Union


class HQNGOptimizer:
    """
    Hamiltonian-aware Quantum Natural Gradient (H-QNG) optimizer
    with optional trust-region (Levenberg–Marquardt) update.

    Update:
        θ' = θ - η (T + λ I)^{-1} ∇L(θ)

    where
        T_ij = (1 / (2 * sqrt(sum_r a_r^2)))
               * sum_r a_r^2
                 ∂_i <P_r> ∂_j <P_r>
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
        """
        Args:
            qnode: an ansatz callable (NOT necessarily a QNode). Must apply gates given theta.
            hamiltonian_terms: qml.Hamiltonian or list of (coeff, Operator)
            eta: stepsize
            lam: damping / trust-region parameter
            lu: accept/reject logic (like WA-QNG)
            rcond: rcond for pseudo-inverse
            diff_method: differentiation method for expval term qnodes
            device: PennyLane device. If None, we will try to infer from cost_fn in step().
            cache_term_qnodes: cache expval qnodes per term for performance
        """
        self.qnode = qnode  # ansatz callable
        self.device = device
        self.eta = eta
        self.lam = lam
        self.lu = lu
        self.lam_factor = 10.0
        self.rcond = rcond
        self.diff_method = diff_method
        self.cache_term_qnodes = cache_term_qnodes

        # Parse Hamiltonian
        if isinstance(hamiltonian_terms, qml.Hamiltonian):
            self.coeffs = np.asarray(hamiltonian_terms.coeffs, dtype=float)
            self.ops = list(hamiltonian_terms.ops)
        else:
            self.coeffs = np.asarray([c for c, _ in hamiltonian_terms], dtype=float)
            self.ops = [op for _, op in hamiltonian_terms]

        if self.coeffs.size == 0:
            raise ValueError("Hamiltonian must contain at least one term.")

        self.norm_factor = 2.0 * np.sqrt(np.sum(self.coeffs ** 2))

        # Optional cache for expval qnodes: key by index
        self._term_qnodes = {}

    # ==========================================================
    # Public API
    # ==========================================================

    def step(self, cost_fn, theta, *args, **kwargs):
        """
        cost_fn is typically your 'energy' QNode.
        We can infer device from cost_fn.device if device wasn't supplied at init.
        """
        theta = np.asarray(theta, dtype=float)

        dev = self._get_device(cost_fn)

        old_cost = float(cost_fn(theta, *args, **kwargs))
        grad = self._gradient(cost_fn, theta, *args, **kwargs)

        T = self._hqng_metric(theta, dev)
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
                    f"[H-QNG] Rejecting step: "
                    f"E_new={new_cost:.6f} >= E_old={old_cost:.6f}. "
                    f"λ → {lam * self.lam_factor:.1e}"
                )
                self.lam = lam * self.lam_factor
                return theta
        else:
            return new_theta

    # ==========================================================
    # Internal helpers
    # ==========================================================

    def _get_device(self, cost_fn) -> qml.devices.Device:
        """
        Prefer explicit device passed at init; otherwise infer from the provided cost QNode.
        """
        if self.device is not None:
            return self.device

        # Try infer from cost_fn if it is a QNode
        dev = getattr(cost_fn, "device", None)
        if dev is None:
            raise ValueError(
                "HQNGOptimizer requires a PennyLane device. "
                "Pass device=... to the optimizer, or pass a QNode cost_fn with a .device attribute."
            )
        return dev

    def _gradient(self, cost_fn, theta, *args, **kwargs):
        grad_fn = qml.grad(cost_fn)
        return np.asarray(grad_fn(theta, *args, **kwargs), dtype=float)

    def _get_term_qnode(self, dev, term_index: int, P_r):
        """
        Build or reuse a QNode that returns <P_r> under the ansatz.
        """
        if self.cache_term_qnodes and term_index in self._term_qnodes:
            return self._term_qnodes[term_index]

        @qml.qnode(dev, interface="autograd", diff_method=self.diff_method)
        def expval_qnode(theta_):
            self.qnode(theta_)
            return qml.expval(P_r)

        if self.cache_term_qnodes:
            self._term_qnodes[term_index] = expval_qnode

        return expval_qnode

    def _hqng_metric(self, theta: np.ndarray, dev) -> np.ndarray:
        """
        Hamiltonian-aware pullback metric (Definition):

            G_ij = sum_r a_r^2 ∂_i<P_r> ∂_j<P_r>
            T_ij = (1 / (2 * sqrt(sum_r a_r^2))) * G_ij
        """
        n_params = theta.size
        G = np.zeros((n_params, n_params), dtype=float)

        for idx, (a_r, P_r) in enumerate(zip(self.coeffs, self.ops)):
            expval_qnode = self._get_term_qnode(dev, idx, P_r)
            grad_pr = qml.grad(expval_qnode)(theta)
            G += (a_r ** 2) * np.outer(grad_pr, grad_pr)

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

    opt = HQNGOptimizer(
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