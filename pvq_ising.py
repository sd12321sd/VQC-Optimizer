import sys
sys.path.insert(0, "./")

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import numpy as onp

from optimizers import PQNGOptimizer

# ============================================================
# Build TFIM (Ising) Hamiltonian
#   H = -J Σ Z_i Z_{i+1} - h Σ X_i
# ============================================================

n_qubits = 4
J = 1.0
h = 1.0
periodic = False

coeffs = []
ops = []

# -J Z_i Z_{i+1}
for i in range(n_qubits - 1):
    coeffs.append(-J)
    ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))

if periodic and n_qubits > 2:
    coeffs.append(-J)
    ops.append(qml.PauliZ(n_qubits - 1) @ qml.PauliZ(0))

# -h X_i
for i in range(n_qubits):
    coeffs.append(-h)
    ops.append(qml.PauliX(i))

H = qml.Hamiltonian(coeffs, ops)
ham_terms = list(zip(H.coeffs, H.ops))

print("Built TFIM Hamiltonian.")
print(f"Number of qubits = {n_qubits} | J = {J} | h = {h} | periodic = {periodic}")

# ============================================================
# Device
# ============================================================

dev = qml.device("default.qubit", wires=n_qubits + 1)

# ============================================================
# EfficientSU2-style Ansatz
#   Each layer:
#       RY on all qubits
#       RZ on all qubits
#       linear CNOT entanglers
#
#   theta is 1D of length = 2 * n_layers * n_qubits
# ============================================================

n_layers = 4

def ansatz(theta):
    theta = qml.math.reshape(theta, (n_layers, 2, n_qubits))

    for l in range(n_layers):
        # single-qubit SU(2)-style rotations
        for i in range(n_qubits):
            qml.RY(theta[l, 0, i], wires=i)
            qml.RZ(theta[l, 1, i], wires=i)

        # linear entanglement
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])


@qml.qnode(dev, interface="autograd")
def energy(theta):
    ansatz(theta)
    return qml.expval(H)

# Optional: final state for overlap/fidelity
@qml.qnode(dev, interface="autograd")
def state(theta):
    ansatz(theta)
    return qml.state()

# ============================================================
# Initialisation
# ============================================================

np.random.seed(123)
theta0 = 0.01 * np.random.randn(2 * n_layers * n_qubits)

theta_qng = theta0.copy()
theta_pqng = theta0.copy()

print("Initial energy:", float(energy(theta0)))

# ============================================================
# Optimizers
# ============================================================

eta = 0.01
lam1 = 1e-3
lam2 = 0.03

# PennyLane built-in QNG
opt_qng = qml.QNGOptimizer(
    stepsize=eta,
    approx=None,
    lam=lam1
)

# Your PQNG
opt_pqng = PQNGOptimizer(
    qnode=ansatz,
    hamiltonian_terms=ham_terms,
    eta=eta,
    lam=lam2,
    lu=True,
    device=dev,
    norm_factor=None,
)

# ============================================================
# Optimization loop
# ============================================================

steps = 40

res_qng = [float(energy(theta_qng))]
res_pqng = [float(energy(theta_pqng))]

print(f"{'Step':>4} | {'QNG':>14} | {'PQNG':>14}")
print("-" * 42)

for s in range(1, steps + 1):
    theta_qng = opt_qng.step(energy, theta_qng)
    theta_pqng = opt_pqng.step(energy, theta_pqng)

    E_q = float(energy(theta_qng))
    E_p = float(energy(theta_pqng))

    res_qng.append(E_q)
    res_pqng.append(E_p)

    if s % 1 == 0 or s == 1 or s == steps:
        print(f"{s:4d} | {E_q:14.8f} | {E_p:14.8f}")

# ============================================================
# Exact ground state
# ============================================================

try:
    H_mat = qml.matrix(H, format="dense")
except TypeError:
    H_mat = qml.matrix(H)

eigvals, eigvecs = onp.linalg.eigh(H_mat)
E_exact = float(onp.min(eigvals))
psi_exact = eigvecs[:, 0]

print("\nExact ground-state energy:", E_exact)
print("Final QNG energy :", res_qng[-1], " | error =", res_qng[-1] - E_exact)
print("Final PQNG energy:", res_pqng[-1], " | error =", res_pqng[-1] - E_exact)


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(7, 5))
plt.axhline(E_exact, color="black", ls="--", lw=2, label="Exact ground state")

plt.plot(res_qng, "-o", ms=3, label="QNG (PennyLane)")
plt.plot(res_pqng, "-o", ms=3, label="PQNG")

plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title(f"TFIM (J={J}, h={h}): QNG vs PQNG with EfficientSU2-style Ansatz")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()