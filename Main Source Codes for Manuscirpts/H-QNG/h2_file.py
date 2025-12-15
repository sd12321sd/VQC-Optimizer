import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import os
import time

num_qubit = 4
dev_tar = qml.device("default.qubit", wires=num_qubit)
dev = qml.device("default.qubit", wires=num_qubit + 1)
initial_params = 3 * np.array([0.7, 0.6, 0.5, 0.1, 0.4], requires_grad=True)

def ansatz(params, wires=[0,1,2,3]):
    qml.RX(params[0], wires=0)
    qml.RX(params[1], wires=1)
    qml.RX(params[2], wires=2)
    qml.RX(params[3], wires=3)
    qml.DoubleExcitation(params[4], wires=[0, 1, 2, 3])
    
@qml.qnode(dev)
def circuit(params, H, num_qubit=num_qubit):
    ansatz(params)
    return qml.expval(H)

h2_data = qml.data.load("qchem", molname="H2", bondlength=0.742, basis="STO-3G")
H = h2_data[0].hamiltonian

H_list = H.terms()[1]
eigen = np.linalg.eigh(H.matrix())
gse = min(eigen[0])
v = len(H_list) - 1

mt_fn = qml.metric_tensor(circuit)

def mt_qng(circuit, params, H):
    mt = mt_fn(params, H)
    return mt

def mt_hqng(circuit, params, H):
    grad_fn = qml.grad(circuit)
    mt = 0
    gradient = 0
    prefactor = 0
    coes, terms = H.terms()
    for k in range(len(terms)):
        coe = coes[k]
        term = terms[k]
        jacobian = qml.math.detach(grad_fn(params, term))
        gradient = gradient + coe * jacobian
        mt = mt + np.outer(jacobian, jacobian) * (coe ** 2)
        prefactor = prefactor + coe * coe
    return  mt / (2 * np.sqrt(prefactor))

params = initial_params
initial_value = qml.math.detach(circuit(params, H)).item()
value_vg = []
value_vg.append(initial_value)
param_listvg = []
param_listvg.append(params)
for i in range(500):
    grad_fn = qml.grad(circuit)
    jacobian = qml.math.detach(grad_fn(params, H))
    params = params -  0.01 * jacobian
    value_vg.append(qml.math.detach(circuit(params, H)).item())
    param_listvg.append(params)
    if i%20 == 0:
        print(i)
value_vg = np.stack(value_vg)


params = initial_params
initial_value = qml.math.detach(circuit(params, H).item())
value_qng = []
value_qng.append(initial_value)
param_listqng = []
param_listqng.append(params)
for i in range(500):
    mt = mt_qng(circuit, params, H)
    grad_fn = qml.grad(circuit)
    jacobian = qml.math.detach(grad_fn(params, H))
    mt_inv = np.linalg.pinv(mt)
    grad = qml.math.detach(np.real(np.dot(mt_inv, jacobian)))
    params = params - 0.01 * grad
    param_listqng.append(params)
    value_qng.append(qml.math.detach(circuit(params, H)).item())
    if i%20 == 0:
        print(i)
value_qng = np.stack(value_qng)


params = initial_params
initial_value = qml.math.detach(circuit(params, H).item())
value_hqng = []
value_hqng.append(initial_value)
param_listhqng = []
param_listhqng.append(params)
for i in range(500):
    mt = mt_hqng(circuit, params, H)
    grad_fn = qml.grad(circuit)
    jacobian = qml.math.detach(grad_fn(params, H))
    I = np.eye(mt.shape[1])
    lam = 0.1
    mt_inv = np.linalg.inv(mt + lam * I)
    grad = qml.math.detach(np.real(np.dot(mt_inv, jacobian)))
    params = params - 0.01 * grad
    param_listhqng.append(params)
    value_hqng.append(qml.math.detach(circuit(params, H)).item())
    if i % 20 == 0:
        print(i)
value_hqng = np.stack(value_hqng)

value_vg_log = np.log10(value_vg - gse + 1e-10)
value_qng_log = np.log10(value_qng - gse + 1e-10)
value_hqng_log = np.log10(value_hqng - gse + 1e-10)

plt.plot(value_vg_log[:350], linewidth=2, label='vg', color='blue')
plt.plot(value_qng_log[:350], linewidth=2, label='qng', color='orange')
plt.plot(value_hqng_log[:350], linewidth=2, label='hqng', color='green')
plt.xlabel("optimization step")
plt.ylabel(r"$\lg(\Delta E)$")
plt.axhline(y=-3, color='red', linestyle='--', linewidth=1)
plt.text(
    x=0.5,         
    y=-3,       
    s="chemical accuracy", 
    color='red',
    va='bottom',   
    ha='left'      
)

plt.title("4-qubit H2 hamiltonian")
plt.minorticks_on()
plt.grid(True)  
plt.legend()







