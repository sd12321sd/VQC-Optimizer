import sys
sys.path.insert(0, "./")

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

from optimizers import PQNGOptimizer

# ============================================================
# Build TFIM (Ising) Hamiltonian
#   H = -J Σ Z_i Z_{i+1} - h Σ X_i
# ============================================================

n_qubits = 4
J = 1.0
h = 0.7
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

dev = qml.device("default.qubit", wires=n_qubits+1)

# ============================================================
# Ansatz
#   2 layers: (RY -> RZ) on each qubit, then CNOT chain
#   theta is 1D of length = 2 * n_layers * n_qubits
# ============================================================

n_layers = 2

def ansatz(theta):
    theta = qml.math.reshape(theta, (n_layers, 2, n_qubits))
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RY(theta[l, 0, i], wires=i)
            qml.RZ(theta[l, 1, i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])


@qml.qnode(dev, interface="autograd")
def energy(theta):
    ansatz(theta)
    return qml.expval(H)

# ============================================================
# Initialisation
# ============================================================

np.random.seed(123)
theta0 = 0.1 * np.random.randn(2 * n_layers * n_qubits)

theta_qng  = theta0.copy()
theta_pqng = theta0.copy()

print("Initial energy:", energy(theta0))

# ============================================================
# Optimizers
# ============================================================

eta = 0.01
lam = 1e-2

opt_qng = qml.QNGOptimizer(stepsize=eta, approx=None)

opt_pqng = PQNGOptimizer(
    qnode=ansatz,
    hamiltonian_terms=ham_terms,
    eta=eta,
    lam=lam,
    lu=False,
    device=dev,
)

# ============================================================
# Optimization loop
# ============================================================

steps = 200

res_qng  = [energy(theta_qng)]
res_pqng = [energy(theta_pqng)]

print(f"{'Step':>4} | {'QNG':>12} | {'P-QNG':>12}")
print("-" * 38)

for s in range(1, steps + 1):
    theta_qng  = opt_qng.step(energy, theta_qng)
    theta_pqng = opt_pqng.step(energy, theta_pqng)

    E_q = energy(theta_qng)
    E_p = energy(theta_pqng)

    res_qng.append(E_q)
    res_pqng.append(E_p)

    if s % 20 == 0 or s == 1:
        print(f"{s:4d} | {E_q:12.8f} | {E_p:12.8f}")

# ============================================================
# Exact ground state
# ============================================================

import numpy as onp

try:
    H_mat = qml.matrix(H, format="dense")
except TypeError:
    H_mat = qml.matrix(H)

eigvals = onp.linalg.eigvalsh(H_mat)
E_exact = float(onp.min(eigvals))

print("\nExact ground-state energy:", E_exact)

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(7, 5))
plt.axhline(E_exact, color="black", ls="--", lw=2, label="Exact ground state")

plt.plot(res_qng, "-o", ms=3, label="QNG")
plt.plot(res_pqng, "-o", ms=3, label="P-QNG")

plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title(f"TFIM (J={J}, h={h}): QNG vs P-QNG")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
