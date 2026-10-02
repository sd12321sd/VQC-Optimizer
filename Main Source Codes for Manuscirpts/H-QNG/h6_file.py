import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as onp
import pennylane as qml
from pennylane import numpy as np


NUM_QUBITS = 12
NUM_STEPS = 100
NUM_RUNS = 10
LEARNING_RATE = 0.01
HQNG_REGULARIZATION = 0.01
INITIALIZATION_LOW = -onp.pi
INITIALIZATION_HIGH = onp.pi
BASE_SEED = 2025
MAX_WORKERS = min(NUM_RUNS, os.cpu_count() or 1)
PROGRESS_EVERY = 1
RESULTS_DIR = Path(__file__).resolve().parent / "results_h6_global"
DATASET_ROOT = Path(os.environ.get("HQNG_DATASET_ROOT", "/data/shic/datasets"))
DATASET_CACHE_DIR = DATASET_ROOT / "h6_standard"


h6 = qml.data.load(
    "qchem",
    molname="H6",
    basis="STO-3G",
    attributes=[
        "fci_energy",
        "hamiltonian",
        "fci_spectrum",
        "hf_state",
        "tapered_spinz_op",
        "vqe_energy",
        "vqe_params",
        "vqe_gates",
    ],
    folder_path=DATASET_CACHE_DIR,
)[0]

FCI_ENERGY = float(h6.fci_energy)
HAMILTONIAN = h6.hamiltonian
HAMILTONIAN_COEFFICIENTS, HAMILTONIAN_TERMS = HAMILTONIAN.terms()
HAMILTONIAN_COEFFICIENTS = onp.asarray(HAMILTONIAN_COEFFICIENTS)
HAMILTONIAN_TERMS = tuple(HAMILTONIAN_TERMS)
HQNG_NORMALIZATION = 2.0 * onp.sqrt(
    onp.sum(HAMILTONIAN_COEFFICIENTS**2)
)

HF_STATE = onp.asarray(h6.hf_state).copy()
NUM_PARAMS = onp.asarray(h6.vqe_params).size
GATE_TEMPLATES = tuple((type(gate), gate.wires) for gate in h6.vqe_gates)
h6.close()
del h6


device = qml.device("default.qubit", wires=NUM_QUBITS + 1)


def ansatz(params, wires=tuple(range(NUM_QUBITS))):
    qml.BasisState(HF_STATE, wires=wires)
    for index, (gate_class, gate_wires) in enumerate(GATE_TEMPLATES):
        gate_class(params[index], wires=gate_wires)


@qml.qnode(device)
def energy_circuit(params):
    ansatz(params)
    return qml.expval(HAMILTONIAN)


@qml.qnode(device)
def single_observable_circuit(params, observable):
    ansatz(params)
    return qml.expval(observable)


ENERGY_GRADIENT_FN = qml.grad(energy_circuit, argnum=0)
QNG_METRIC_FN = qml.metric_tensor(energy_circuit, approx="block-diag")
SINGLE_TERM_GRADIENT_FN = qml.grad(single_observable_circuit, argnum=0)


def as_numpy(value):
    return onp.asarray(qml.math.toarray(qml.math.detach(value)))


def as_trainable(values):

    return np.array(values, requires_grad=True)


def evaluate_energy(params):
    return float(as_numpy(energy_circuit(params)))


def compute_hqng_quantities(params):


    gradient = onp.zeros(NUM_PARAMS, dtype=float)
    metric = onp.zeros((NUM_PARAMS, NUM_PARAMS), dtype=float)

    for coefficient, term in zip(
        HAMILTONIAN_COEFFICIENTS,
        HAMILTONIAN_TERMS,
    ):
        term_gradient = onp.real(
            as_numpy(SINGLE_TERM_GRADIENT_FN(params, term))
        )
        gradient += coefficient * term_gradient
        metric += coefficient**2 * onp.outer(term_gradient, term_gradient)

    metric /= HQNG_NORMALIZATION
    return onp.real(gradient), onp.real(metric)


def initial_parameters(run_index):


    rng = onp.random.default_rng(BASE_SEED + run_index)
    return rng.uniform(
        INITIALIZATION_LOW,
        INITIALIZATION_HIGH,
        size=NUM_PARAMS,
    )


def report_progress(run_index, method, completed_steps):
    if completed_steps % PROGRESS_EVERY == 0 or completed_steps == NUM_STEPS:
        print(
            f"run {run_index:02d} | {method:<5} | "
            f"step {completed_steps:03d}/{NUM_STEPS}",
            flush=True,
        )


def run_vg(initial_params, run_index):
    params = as_trainable(initial_params)
    energies = onp.empty(NUM_STEPS + 1)
    energies[0] = evaluate_energy(params)

    for step in range(1, NUM_STEPS + 1):
        gradient = onp.real(as_numpy(ENERGY_GRADIENT_FN(params)))
        params = as_trainable(as_numpy(params) - LEARNING_RATE * gradient)
        energies[step] = evaluate_energy(params)
        report_progress(run_index, "VG", step)

    return energies


def run_qng(initial_params, run_index):
    params = as_trainable(initial_params)
    energies = onp.empty(NUM_STEPS + 1)
    energies[0] = evaluate_energy(params)

    for step in range(1, NUM_STEPS + 1):
        metric = onp.real(as_numpy(QNG_METRIC_FN(params)))
        gradient = onp.real(as_numpy(ENERGY_GRADIENT_FN(params)))

        direction = onp.linalg.pinv(metric) @ gradient
        params = as_trainable(as_numpy(params) - LEARNING_RATE * direction)
        energies[step] = evaluate_energy(params)
        report_progress(run_index, "QNG", step)

    return energies


def run_hqng(initial_params, run_index):
    params = as_trainable(initial_params)
    identity = onp.eye(initial_params.size)
    energies = onp.empty(NUM_STEPS + 1)
    energies[0] = evaluate_energy(params)

    for step in range(1, NUM_STEPS + 1):
        gradient, metric = compute_hqng_quantities(params)
        regularized_metric = metric + HQNG_REGULARIZATION * identity

        direction = onp.linalg.solve(regularized_metric, gradient)
        params = as_trainable(as_numpy(params) - LEARNING_RATE * direction)
        energies[step] = evaluate_energy(params)
        report_progress(run_index, "H-QNG", step)

    return energies


def process_run(run_index):


    initial_params = initial_parameters(run_index)
    print(f"run {run_index:02d} | started", flush=True)
    onp.save(RESULTS_DIR / f"initial_params{run_index}.npy", initial_params)

    print(f"run {run_index:02d} | H-QNG | started", flush=True)
    value_hqng = run_hqng(initial_params, run_index)
    onp.save(RESULTS_DIR / f"value_hqng{run_index}.npy", value_hqng)

    print(f"run {run_index:02d} | VG | started", flush=True)
    value_vg = run_vg(initial_params, run_index)
    onp.save(RESULTS_DIR / f"value_vg{run_index}.npy", value_vg)

    print(f"run {run_index:02d} | QNG | started", flush=True)
    value_qng = run_qng(initial_params, run_index)
    onp.save(RESULTS_DIR / f"value_qng{run_index}.npy", value_qng)

    print(f"run {run_index:02d} | completed", flush=True)
    return run_index


def main():
    start_time = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"H6 FCI energy: {FCI_ENERGY}")
    print(f"Number of parameters: {NUM_PARAMS}")
    print(f"Number of Hamiltonian terms: {len(HAMILTONIAN_TERMS)}")
    print(
        "Initialization: independent Uniform"
        f"([{INITIALIZATION_LOW:.6f}, {INITIALIZATION_HIGH:.6f}])"
    )
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Running {NUM_RUNS} trials with {MAX_WORKERS} worker(s).")

    if NUM_RUNS == 1:
        process_run(0)
    else:
        failures = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_run, run_index): run_index
                for run_index in range(NUM_RUNS)
            }
            for future in as_completed(futures):
                run_index = futures[future]
                try:
                    future.result()
                except Exception as error:
                    failures.append((run_index, error))
                    print(f"run {run_index:02d} | failed: {error}", flush=True)

        if failures:
            failed_runs = ", ".join(str(index) for index, _ in failures)
            raise RuntimeError(f"H6 experiment failed for run(s): {failed_runs}")

    elapsed = time.perf_counter() - start_time
    print(f"Total time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
