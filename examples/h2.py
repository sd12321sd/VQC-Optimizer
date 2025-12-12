import sys
sys.path.insert(0, "./")

import pennylane as qml
from pennylane import numpy as np
from optimizers import WAQNGOptimizer
import matplotlib.pyplot as plt

h2_data = qml.data.load("qchem", molname="H2", bondlength=0.742, basis="STO-3G")
H = h2_data[0].hamiltonian
n_qubits = H.num_wires

print("Loaded H2 Hamiltonian from qml.data.load.")
print(f"Number of qubits = {n_qubits}")

ham_terms = list(zip(H.coeffs, H.ops))


def ansatz(params, wires=[0,1,2,3]):
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.RX(params[2], wires=wires[2])
    qml.RX(params[3], wires=wires[3])
    qml.DoubleExcitation(params[4], wires=wires)


dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="autograd")
def energy(theta):
    ansatz(theta)
    return qml.expval(H)

_orig_dm = qml.density_matrix

def _patched_dm(wires, *args, **kwargs):
    if isinstance(wires, (list, tuple)) and len(wires) == 0:
        wires = dev.wires
    return _orig_dm(wires, *args, **kwargs)

qml.density_matrix = _patched_dm


np.random.seed(123)
theta0 = np.random.randn(5) * 0.1

theta_qng = theta0.copy()
theta_waqng = theta0.copy()

print("Initial energy:", energy(theta0))

eta = 1e-2
opt_qng = qml.QNGOptimizer(stepsize=eta)

opt_waqng = WAQNGOptimizer(
    qnode=ansatz,
    hamiltonian_terms=ham_terms,
    eta=eta,
    lam=1e-5,
    lu=True,        
    device=dev
)


steps = 200

res_qng = [energy(theta_qng)]
res_waqng = [energy(theta_waqng)]

print(f"{'Step':>4} | {'E(QNG)':>14} | {'E(WA-QNG)':>14}")
print("-" * 45)

for s in range(1, steps+1):
    theta_qng = opt_qng.step(energy, theta_qng)
    E_q = energy(theta_qng)

    theta_waqng = opt_waqng.step(energy, theta_waqng)
    E_w = energy(theta_waqng)

    res_qng.append(E_q)
    res_waqng.append(E_w)

    print(f"{s:4d} | {E_q:14.8f} | {E_w:14.8f}")


import numpy as onp
try:
    H_mat = qml.matrix(H, format="dense")
except TypeError:
    H_mat = qml.matrix(H)
eigvals = onp.linalg.eigvalsh(H_mat)
E_exact = float(np.min(eigvals))
print("\nExact ground-state energy:", E_exact)

plt.figure(figsize=(7,5))
plt.axhline(E_exact, color="gray", ls="--", lw=2, label="Exact ground state")
plt.plot(res_qng, "-o", label="QNG")
plt.plot(res_waqng, "-o", label="WA-QNG")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("H₂ with simple ansatz")
plt.grid()
plt.legend()
plt.show()

    


