import pennylane as qml
from pennylane import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def ising_observable(num_qubit):
    observable = 0
    for i in range(num_qubit):
        observable = observable + qml.PauliX(i)
    for i in range(num_qubit-1):
        observable = observable + qml.PauliZ(i) @ qml.PauliZ(i+1)
    return observable

def hesen_observable(num_qubit):
    observable = 0
    for i in range(num_qubit-1):
        observable = observable + qml.PauliZ(i) @ qml.PauliZ(i+1) + qml.PauliX(i) @ qml.PauliX(i+1) + qml.PauliY(i) @ qml.PauliY(i+1)
    return observable


run_start = 0
def process(run):
    
    np.random.seed(run)
    
    flag = 'ising'
    num_qubit = 4
    num_layer = 2
    rec_wa = 1e-3 #defaut 1e-3

    
    initial_params = np.random.uniform(0.05, 0.1, 2 * num_qubit * num_layer)
    for i in range(len(initial_params)):
        initial_params[i] = initial_params[i] * (2 * np.random.randint(0, 2) - 1)
    
    
    initial_params = initial_params[:2*num_qubit*num_layer]
    
    lr = 0.02
    num_epoch = 500
    
    if flag == 'ising':
        observable = ising_observable(num_qubit)
    else:
        observable = hesen_observable(num_qubit)
    
    ob_coe, ob_list = observable.terms()
    ob_sub = [list(term.wires) for term in ob_list]
    ob_length = len(ob_list)
    
    shift = np.pi / 2
    dev = qml.device("default.qubit", wires=num_qubit)
    
    def circuit(params, num_qubit=num_qubit, num_layer=num_layer):
        for i in range(num_layer):
            for j in range(num_qubit):
                qml.RX(params[2 * num_qubit * i + 2 * j], wires=j)
                qml.RY(params[2 * num_qubit * i + 2 * j + 1], wires=j)
            for j in range(num_qubit-1):
                qml.CNOT(wires=[j+1,j]) 
    
    @qml.qnode(dev)
    def circuit_exp(params, num_qubit=num_qubit, num_layer=num_layer, observable=observable):
        circuit(params, num_qubit=num_qubit, num_layer=num_layer)
        return qml.expval(observable)
    
    @qml.qnode(dev)
    def circuit_fullspace(params, num_qubit=num_qubit, num_layer=num_layer):
        circuit(params, num_qubit=num_qubit, num_layer=num_layer)
        return qml.density_matrix(wires=[i for i in range(num_qubit)])
    
    @qml.qnode(dev)
    def circuit_subspace(params, num_qubit=num_qubit, num_layer=num_layer, ob_sub=ob_sub):
        circuit(params, num_qubit=num_qubit, num_layer=num_layer)
        return qml.density_matrix(wires=ob_sub)
    
    def state_to_density(state_vector):
        return np.outer(state_vector, np.conj(state_vector))
    
    def t_element_compute(partial1, partial2):
        #print(np.abs(partial1))
        return 2 * np.trace(np.dot(partial1, partial2))
    
    def circuit(params, num_qubit=num_qubit, num_layer=num_layer):
        for i in range(num_layer):
            for j in range(num_qubit):
                qml.RX(params[2 * num_qubit * i + 2 * j], wires=j)
                qml.RY(params[2 * num_qubit * i + 2 * j + 1], wires=j)
            for j in range(num_qubit-1):
                qml.CNOT(wires=[j+1,j]) 
    
    def f_matrix_compute(circuit, params):
        shift = np.pi / 2
        densities = []
        for idx in range(len(params)):
            shifted_params = params.copy()
            shifted_params[idx] += shift
            forward_state = circuit(shifted_params)
            shifted_params = params.copy()
            shifted_params[idx] -= shift
            backward_state = circuit(shifted_params)
            densities.append( qml.math.detach((forward_state - backward_state) / (2 * np.sin(shift)) ))
        T_matrix = np.zeros((len(params),len(params)),dtype = 'complex_')
        for i in range(len(params)):
            for j in range(len(params)):
                T_matrix[i][j] = t_element_compute(densities[i], densities[j])
        return T_matrix
    
    def t_matrix_compute(circuit, params, ob_coe, ob_sub):
        shift = np.pi / 2
        T_matrix_full = 0
        for iob in range(ob_length):
            densities = []
            for idx in range(len(params)):
                shifted_params = params.copy()
                shifted_params[idx] += shift
                forward_state = circuit(shifted_params, ob_sub = ob_sub[iob])
                shifted_params = params.copy()
                shifted_params[idx] -= shift
                backward_state = circuit(shifted_params, ob_sub = ob_sub[iob])
                densities.append( qml.math.detach((forward_state - backward_state) / (2 * np.sin(shift)) ))
            T_matrix = np.zeros((len(params),len(params)),dtype = 'complex_')
            for i in range(len(params)):
                for j in range(len(params)):
                    if np.abs(i-j) <= len(params) / num_layer:
                        T_matrix[i][j] = t_element_compute(densities[i], densities[j])
            T_matrix_full = T_matrix_full + ob_coe[iob]**2 * T_matrix
        T_matrix_full = T_matrix_full / np.square(ob_coe).sum()
        return T_matrix_full
    
    def t_matrix_compute(circuit, params, ob_coe, ob_sub):
        shift = np.pi / 2
        T_matrix_full = 0
        for iob in range(ob_length):
            densities = []
            for idx in range(len(params)):
                shifted_params = params.copy()
                shifted_params[idx] += shift
                forward_state = circuit(shifted_params, ob_sub = ob_sub[iob])
                shifted_params = params.copy()
                shifted_params[idx] -= shift
                backward_state = circuit(shifted_params, ob_sub = ob_sub[iob])
                densities.append( qml.math.detach((forward_state - backward_state) / (2 * np.sin(shift)) ))
            T_matrix = np.zeros((len(params),len(params)),dtype = 'complex_')
            for i in range(len(params)):
                for j in range(len(params)):
                    if np.abs(i-j) <= len(params) / num_layer:
                        T_matrix[i][j] = t_element_compute(densities[i], densities[j])
            T_matrix_full = T_matrix_full + ob_coe[iob]**2 * T_matrix
        T_matrix_full = T_matrix_full / np.square(ob_coe).sum()
        return T_matrix_full
    
    def qng(circuit_fullspace, circuit_exp, params, rec=1e-8):
        T_matrix = f_matrix_compute(circuit_fullspace, params)
        pinv = np.linalg.pinv(T_matrix)
        grad_fn = qml.grad(circuit_exp)
        jacobian = grad_fn(params)
        result = qml.math.detach(np.real(np.dot(pinv, jacobian)))
        return result
    
    def wa_qng(circuit_subspace, circuit_exp, params, ob_coe, ob_sub, rec=rec_wa):
        T_matrix = t_matrix_compute(circuit_subspace, params, ob_coe, ob_sub)
        lambda_reg = rec
        I = np.eye(T_matrix.shape[1]) 
        pinv = np.linalg.inv(T_matrix.conj().T @ T_matrix + lambda_reg * I) @ T_matrix.conj().T
        grad_fn = qml.grad(circuit_exp)
        jacobian = grad_fn(params)
        result = qml.math.detach(np.real(np.dot(pinv, jacobian)))
        return result
    
    
    t0 = time.time()
    print("vanilla gradient training starts") 
    opt = qml.GradientDescentOptimizer(lr/4)
    params = initial_params
    initial_value = qml.math.detach(circuit_exp(params)).item()
    value_vg = []
    value_vg.append(initial_value)
    param_listvg = []
    param_listvg.append(params)
    for i in range(num_epoch):
        params = opt.step(circuit_exp, params)
        value_vg.append(circuit_exp(params))
        print(f"training process:{i/num_epoch * 100}%_run_{run}")
    print("vanilla gradient training ends") 
    
    t1 = time.time()
    
    print("wa natural gradient training starts") 
    params = initial_params
    initial_value = qml.math.detach(circuit_exp(params)).item()
    value_ang = []
    value_ang.append(initial_value)
    for i in range(num_epoch):
        nqg = wa_qng(circuit_subspace, circuit_exp, params, ob_coe, ob_sub)
        params = params - lr * nqg
        value_ang.append(circuit_exp(params))
        print(f"training process:{i/num_epoch * 100}%_run_{run}")
    print("wa natural gradient training ends")   
    
    t2 = time.time()
    
    opt = qml.QNGOptimizer(lr/4, lam=1e-3)
    
    
    print("natural gradient training starts")
    params = initial_params
    initial_value = qml.math.detach(circuit_exp(params)).item()
    value_ng = []
    value_ng.append(initial_value)
    param_listng = []
    param_listng.append(params)
    for i in range(num_epoch):
        params = opt.step(circuit_exp, params)
        value_ng.append(circuit_exp(params))
        print(f"training process:{i/num_epoch * 100}%_run_{run}")
    print("natural gradient training ends")
    
    t3 = time.time()
    
    print(f"Vanilla NG using time: {t1-t0}")
    print(f'WAQNG using time: {t2-t1}')
    print(f'QNG using time: {t3-t2}')
    
    value_vg = np.array(value_vg)
    value_ng = np.array(value_ng)
    value_ang = np.array(value_ang)
    
    folder_name = f"results_nq{num_qubit}_nl{num_layer}_wa{rec_wa}" + flag

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)  

    
    np.save(folder_name + f"/value_vg_{run}.npy", value_vg)
    np.save(folder_name + f"/value_ng_{run}.npy", value_ng)
    np.save(folder_name + f"/value_ang_{run}.npy", value_ang)



if __name__ == "__main__":
    for i in range(1):
        runs = list(range(run_start+25*i, run_start+25*i+25))
        runs = list(range(2))
            
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process, run): run for run in runs}
    
            for future in as_completed(futures):
                run = futures[future]
                try:
                    result = future.result()
                    print(result)  # 打印每次任务完成的信息
                except Exception as e:
                    print(f"Run {run} generated an exception: {e}")












