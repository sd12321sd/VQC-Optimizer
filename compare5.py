import sys
sys.path.insert(0, "./")

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

from optimizers import PQNGOptimizer, AQNGOptimizer

# ============================================================
# Load H2 Hamiltonian
# ============================================================

h2_data = qml.data.load(
    "qchem",
    molname="H2",
    bondlength=0.742,
    basis="STO-3G"
)

H = h2_data[0].hamiltonian
ham_terms = list(zip(H.coeffs, H.ops))

# STO-3G H2 → 4 qubits
n_qubits = 4

print("Loaded H2 Hamiltonian from qml.data.load.")
print(f"Number of qubits = {n_qubits}")

# ============================================================
# Device
#   Keep one extra wire free (helps metric_tensor in some PennyLane versions)
# ============================================================

dev = qml.device("default.qubit", wires=n_qubits + 1)

# ============================================================
# Ansatz
# ============================================================

def ansatz(theta):
    for i in range(4):
        qml.RX(theta[i], wires=i)
    qml.DoubleExcitation(theta[4], wires=[0, 1, 2, 3])


@qml.qnode(dev, interface="autograd")
def energy(theta):
    ansatz(theta)
    return qml.expval(H)

# ============================================================
# Initialisation
# ============================================================

np.random.seed(123)
theta0 = 0.1 * np.random.randn(5)

theta_pqng = theta0.copy()
theta_aqng = theta0.copy()

print("Initial energy:", energy(theta0))

# ============================================================
# Optimizers
# ============================================================

eta = 0.01
lam = 1e-2
mix = 0.5  # A-QNG mixing: 0 -> P-QNG, 1 -> QNG

opt_pqng = PQNGOptimizer(
    qnode=ansatz,
    hamiltonian_terms=ham_terms,
    eta=eta,
    lam=lam,
    lu=False,
    device=dev,
)

opt_aqng = AQNGOptimizer(
    qnode=ansatz,
    hamiltonian_terms=ham_terms,
    eta=eta,
    lam=lam,
    mix=mix,
    lu=False,
    device=dev,
    approx_qng=None,  # match your compare style; change to "block-diag" if you want
)

# ============================================================
# Optimization loop
# ============================================================

steps = 250

res_pqng = [energy(theta_pqng)]
res_aqng = [energy(theta_aqng)]

print(f"{'Step':>4} | {'P-QNG':>12} | {'A-QNG':>12}")
print("-" * 38)

for s in range(1, steps + 1):
    theta_pqng = opt_pqng.step(energy, theta_pqng)
    theta_aqng = opt_aqng.step(energy, theta_aqng)

    E_p = energy(theta_pqng)
    E_a = energy(theta_aqng)

    res_pqng.append(E_p)
    res_aqng.append(E_a)

    if s % 20 == 0 or s == 1:
        print(f"{s:4d} | {E_p:12.8f} | {E_a:12.8f}")

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

plt.plot(res_pqng, "-o", ms=3, label="P-QNG")
plt.plot(res_aqng, "-o", ms=3, label=f"A-QNG (mix={mix})")

plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("H₂ (STO-3G): P-QNG vs A-QNG")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
