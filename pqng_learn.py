import sys
sys.path.insert(0, "./")

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from optimizers import PQNGOptimizer


# -----------------------------
# 1) Problem: 4-qubit TFIM Hamiltonian
#    H = - sum_i Z_i Z_{i+1} - h * sum_i X_i
# -----------------------------
def make_tfim_hamiltonian(n_qubits=4, h=1.0, periodic=False):
    coeffs = []
    ops = []

    # - Z_i Z_{i+1}
    for i in range(n_qubits - 1):
        coeffs.append(-1.0)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))

    if periodic and n_qubits > 2:
        coeffs.append(-1.0)
        ops.append(qml.PauliZ(n_qubits - 1) @ qml.PauliZ(0))

    # - h X_i
    for i in range(n_qubits):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))

    return qml.Hamiltonian(coeffs, ops)


# -----------------------------
# 2) Ansatz: Hardware-efficient
#    IMPORTANT: theta is now 1D of length (n_layers*n_qubits).
# -----------------------------
def ansatz(theta, n_qubits, n_layers):
    theta = qml.math.reshape(theta, (n_layers, n_qubits))

    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RY(theta[l, i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])


# -----------------------------
# 3) Build QNode + cost function
# -----------------------------
def build_cost(H, n_qubits=4, n_layers=3, shots=None):
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    @qml.qnode(dev, interface="autograd")
    def energy(theta):
        ansatz(theta, n_qubits, n_layers)
        return qml.expval(H)

    return energy, dev


# -----------------------------
# 4) One run of P-QNG optimization for given eta
# -----------------------------
def run_pqng(cost_fn, dev, ham_terms, theta0, eta, steps, lam=1e-6, lu=False):
    """
    P-QNG optimizer run for a fixed stepsize eta.

    lam: trust-region damping for (T + lam I)^{-1}
    lu: accept/reject logic (usually keep False for eta sweeps)
    """
    opt = PQNGOptimizer(
        qnode=lambda th: ansatz(th, n_qubits=4, n_layers=3),  # uses 1D theta; reshapes internally
        hamiltonian_terms=ham_terms,
        eta=float(eta),
        lam=float(lam),
        lu=lu,
        device=dev,
    )

    theta = pnp.array(theta0, requires_grad=True)
    energies = [float(cost_fn(theta))]

    for _ in range(steps):
        theta = opt.step(cost_fn, theta)
        energies.append(float(cost_fn(theta)))

    return np.array(energies), theta


# -----------------------------
# 5) Sweep etas and pick best
# -----------------------------
def find_best_eta_pqng(cost_fn, dev, ham_terms, theta0, eta_grid, steps,
                       lam=1e-6, lu=False, n_restarts=1, seed=0):
    """
    Define 'best eta' = smallest mean final energy after `steps` iterations.
    If n_restarts > 1, re-run from different random initializations (robustness).
    """
    best_eta = None
    best_final = np.inf
    best_curve = None

    rng = np.random.default_rng(seed)

    for eta in eta_grid:
        finals = []
        curves = []

        for r in range(n_restarts):
            if n_restarts == 1:
                theta_init = pnp.array(theta0, requires_grad=True)
            else:
                theta_init = pnp.array(
                    0.1 * rng.standard_normal(size=theta0.shape),
                    requires_grad=True
                )

            curve, _ = run_pqng(
                cost_fn=cost_fn,
                dev=dev,
                ham_terms=ham_terms,
                theta0=theta_init,
                eta=eta,
                steps=steps,
                lam=lam,
                lu=lu
            )
            curves.append(curve)
            finals.append(curve[-1])

        mean_final = float(np.mean(finals))
        print(f"eta={eta: .4e} | mean final E={mean_final: .10f}")

        if mean_final < best_final:
            best_final = mean_final
            best_eta = float(eta)
            best_curve = np.mean(curves, axis=0)

    return best_eta, best_final, best_curve


def main():
    # Experiment settings
    n_qubits = 4
    n_layers = 3
    steps = 80
    h = 1.0
    periodic = False

    # Build problem
    H = make_tfim_hamiltonian(n_qubits=n_qubits, h=h, periodic=periodic)
    cost_fn, dev = build_cost(H, n_qubits=n_qubits, n_layers=n_layers, shots=None)

    # Hamiltonian terms for P-QNG (coeffs, ops) -- coeffs unused by P-QNG but ok to pass
    ham_terms = list(zip(H.coeffs, H.ops))

    # Initial parameters: 1D vector length (n_layers*n_qubits)
    np.random.seed(0)
    theta0 = pnp.array(0.1 * np.random.randn(n_layers * n_qubits), requires_grad=True)

    # Learning-rate grid
    eta_grid = np.logspace(-3, 0, 13)  # 1e-3 ... 1e0

    # P-QNG settings
    lam = 1e-2   # trust-region damping; increase if unstable (e.g., 1e-3)
    lu = False   # keep False for fair sweep

    print("\n=== Sweeping learning rates for P-QNG (1D theta) ===")
    best_eta, best_final, best_curve = find_best_eta_pqng(
        cost_fn=cost_fn,
        dev=dev,
        ham_terms=ham_terms,
        theta0=theta0,
        eta_grid=eta_grid,
        steps=steps,
        lam=lam,
        lu=lu,
        n_restarts=1,
        seed=0
    )

    print("\n=== Result ===")
    print(f"Best eta: {best_eta:.4g}")
    print(f"Best final energy (mean over restarts): {best_final:.10f}")

    print("\nBest curve (E vs step):")
    print("  first 5:", np.round(best_curve[:5], 6))
    print("  last  5:", np.round(best_curve[-5:], 6))

    # Optional plot
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(best_curve)
        plt.xlabel("Step")
        plt.ylabel("Energy")
        plt.title(f"P-QNG best eta={best_eta:.3g} (TFIM {n_qubits}q, h={h})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()