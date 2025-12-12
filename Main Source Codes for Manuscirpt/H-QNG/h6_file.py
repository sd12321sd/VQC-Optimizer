import pennylane as qml
from pennylane import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

t1 = time.time()

num_qubit = 12
dev = qml.device("default.qubit", wires=num_qubit + 1)


h6 = qml.data.load("qchem", molname="H6", basis="STO-3G", attributes=["fci_energy", "fci_spectrum", "hamiltonian", "hf_state", "tapered_spinz_op", "vqe_energy", "vqe_gates", "vqe_params"])
h6 = h6[0]
H = h6.hamiltonian

H_list = H.terms()[1]
v = len(H_list) - 1

gate_templates = []
initial_params = []

folder_name = f"results_h6"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)  



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
    return  gradient, mt / (2 * np.sqrt(prefactor))

for gate in h6.vqe_gates:
    gate_templates.append((type(gate), gate.wires))
    initial_params.append(gate.parameters[0])  # excitation 门只有一个参数
    
hf_state = h6.hf_state

def ansatz(params, wires=list(range(12))):
    qml.BasisState(hf_state, wires)
    for i, (GateClass, wires) in enumerate(gate_templates):
        GateClass(params[i], wires=wires)

@qml.qnode(dev)
def circuit(params, H, num_qubit=num_qubit):
    ansatz(params)
    return qml.expval(H)

mt_fn = qml.metric_tensor(circuit, approx="block-diag")

def process(j):
    np.random.seed() 
    initial_params = h6.vqe_params + np.random.uniform(-0.1, 0.1, len(h6.vqe_params))
  
    print('vg start')
    params = initial_params
    initial_value = qml.math.detach(circuit(params, H)).item()
    value_vg = []
    value_vg.append(initial_value)
    param_listvg = []
    param_listvg.append(params)
    for i in range(1):
        grad_fn = qml.grad(circuit)
        jacobian = qml.math.detach(grad_fn(params, H))
        params = params -  0.01 * jacobian
        value_vg.append(qml.math.detach(circuit(params, H)).item())
        param_listvg.append(params)
        if i%5 == 0:
            print(i)
    value_vg = np.stack(value_vg)
    np.save(folder_name + f"/value_vg{j}.npy", value_vg)
    
    
    print('qng start')
    
    params = initial_params
    initial_value = qml.math.detach(circuit(params, H).item())
    value_qng = []
    value_qng.append(initial_value)
    param_listqng = []
    param_listqng.append(params)
    for i in range(1):
        mt = mt_qng(circuit, params, H)
        grad_fn = qml.grad(circuit)
        jacobian = qml.math.detach(grad_fn(params, H))
        mt_inv = np.linalg.pinv(mt)
        grad = qml.math.detach(np.real(np.dot(mt_inv, jacobian)))
        params = params - 0.01 * grad
        param_listqng.append(params)
        value_qng.append(qml.math.detach(circuit(params, H)).item())
        if i%5 == 0:
            print(i)
    value_qng = np.stack(value_qng)
    np.save(folder_name + f"/value_qng{j}.npy", value_qng)
    
    
    print('hqng start')
    params = initial_params
    initial_value = qml.math.detach(circuit(params, H).item())
    value_hqng = []
    value_hqng.append(initial_value)
    param_listhqng = []
    param_listhqng.append(params)
    for i in range(1):
        gradient, mt = mt_hqng(circuit, params, H)
        I = np.eye(mt.shape[1])
        lam = 0.1
        mt_inv = np.linalg.inv(mt + lam * I)
        grad = qml.math.detach(np.real(np.dot(mt_inv, gradient)))
        params = params - 0.01 * grad
        param_listhqng.append(params)
        value_hqng.append(qml.math.detach(circuit(params, H)).item())
        if i % 5 == 0:
            print(i)
    value_hqng = np.stack(value_hqng)
    np.save(folder_name + f"/value_hqng{j}.npy", value_hqng)

if __name__ == "__main__":
    runs = list(range(1))
        
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process, run): run for run in runs}

        for future in as_completed(futures):
            run = futures[future]
            try:
                result = future.result()
                print(result)  
            except Exception as e:
                print(f"Run {run} generated an exception: {e}")


t2= time.time()

print('total time:')
print(t2-t1)




