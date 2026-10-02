import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as onp
import pennylane as qml
from pennylane import numpy as np


MOLECULE = "LiH"
SCRIPT_DIR = Path(__file__).resolve().parent


def positive_int_from_env(name, default):
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive; got {value}.")
    return value


def results_directory():
    configured = os.environ.get("HQNG_RESULTS_DIR")
    layer_suffix = f"_L{NUM_LAYERS}"
    path = Path(configured).expanduser() if configured else (
        SCRIPT_DIR
        / f"results_{MOLECULE.lower()}_su2{layer_suffix}_hfcenter_{NUM_STEPS}steps"
    )
    return path.resolve() if path.is_absolute() else (SCRIPT_DIR / path).resolve()


NUM_STEPS = positive_int_from_env("HQNG_NUM_STEPS", 100)
NUM_RUNS = positive_int_from_env("HQNG_NUM_RUNS", 10)
NUM_LAYERS = positive_int_from_env("HQNG_NUM_LAYERS", 1)
LEARNING_RATE = float(os.environ.get("HQNG_LEARNING_RATE", "0.01"))
if not onp.isfinite(LEARNING_RATE) or LEARNING_RATE <= 0.0:
    raise ValueError("HQNG_LEARNING_RATE must be finite and positive.")
HQNG_REGULARIZATION = float(os.environ.get("HQNG_REGULARIZATION", "0.1"))
if not onp.isfinite(HQNG_REGULARIZATION) or HQNG_REGULARIZATION <= 0.0:
    raise ValueError("HQNG_REGULARIZATION must be finite and positive.")
BASE_SEED = 2025
INITIALIZATION_CENTER = os.environ.get("HQNG_INIT_CENTER", "hf").strip().lower()
if INITIALIZATION_CENTER not in {"hf", "zero"}:
    raise ValueError("HQNG_INIT_CENTER must be 'hf' or 'zero'.")
INITIALIZATION_LOW = float(os.environ.get("HQNG_INIT_LOW", "-0.1"))
INITIALIZATION_HIGH = float(os.environ.get("HQNG_INIT_HIGH", "0.1"))
if INITIALIZATION_LOW >= INITIALIZATION_HIGH:
    raise ValueError("HQNG_INIT_LOW must be less than HQNG_INIT_HIGH.")
REQUESTED_MAX_WORKERS = positive_int_from_env("HQNG_MAX_WORKERS", NUM_RUNS)
MAX_WORKERS = min(NUM_RUNS, REQUESTED_MAX_WORKERS, os.cpu_count() or 1)
VALIDATE_ONLY = os.environ.get("HQNG_VALIDATE_ONLY", "0").strip().lower() in {
    "1", "true", "yes",
}
PROGRESS_EVERY = 1
GRADIENT_CHECK_EPSILON = 1.0e-6
GRADIENT_CHECK_ATOL = 1.0e-5
GRADIENT_CHECK_RTOL = 1.0e-3
RESULTS_DIR = results_directory()
DATASET_ROOT = Path(os.environ.get("HQNG_DATASET_ROOT", "/data/shic/datasets"))
DATASET_CACHE_DIR = DATASET_ROOT / f"{MOLECULE.lower()}_su2"


dataset_options = {
    "molname": MOLECULE,
    "basis": "STO-3G",
    "attributes": ["fci_energy", "hamiltonian", "hf_state"],
    "folder_path": DATASET_CACHE_DIR,
}
if MOLECULE == "LiH":
    dataset_options["bondlength"] = 1.57

dataset_file = os.environ.get("HQNG_DATASET_FILE")
if dataset_file:
    dataset = qml.data.Dataset.open(dataset_file, mode="r")
else:
    dataset = qml.data.load("qchem", **dataset_options)[0]
FCI_ENERGY = float(dataset.fci_energy)
HAMILTONIAN = dataset.hamiltonian
HF_STATE = onp.asarray(dataset.hf_state, dtype=int).copy()
dataset.close()
del dataset

HAMILTONIAN_COEFFICIENTS, HAMILTONIAN_TERMS = HAMILTONIAN.terms()
HAMILTONIAN_COEFFICIENTS = onp.asarray(HAMILTONIAN_COEFFICIENTS)
if onp.max(onp.abs(onp.imag(HAMILTONIAN_COEFFICIENTS))) > 1.0e-10:
    raise ValueError("The Hamiltonian has non-real Pauli coefficients.")
HAMILTONIAN_COEFFICIENTS = onp.real(HAMILTONIAN_COEFFICIENTS).astype(float)
HAMILTONIAN_TERMS = tuple(HAMILTONIAN_TERMS)
HQNG_NORMALIZATION = 2.0 * onp.sqrt(
    onp.sum(HAMILTONIAN_COEFFICIENTS**2)
)

NUM_QUBITS = int(HF_STATE.size)
if NUM_QUBITS != 12:
    raise ValueError(f"Expected 12 qubits for {MOLECULE}; got {NUM_QUBITS}.")
TARGET_PARTICLE_NUMBER = int(HF_STATE.sum())
PARAM_SHAPE = (NUM_LAYERS, NUM_QUBITS, 2)
NUM_PARAMS = int(onp.prod(PARAM_SHAPE))
TERM_MATRICES = tuple(
    term.sparse_matrix(wire_order=range(NUM_QUBITS)).tocsr()
    for term in HAMILTONIAN_TERMS
)
HAMILTONIAN_MATRIX = HAMILTONIAN.sparse_matrix(
    wire_order=range(NUM_QUBITS)
).tocsr()


def wire_spec(wire):
    permutation = tuple(w for w in range(NUM_QUBITS) if w != wire) + (wire,)
    return permutation, tuple(onp.argsort(permutation))


WIRE_SPECS = tuple(wire_spec(wire) for wire in range(NUM_QUBITS))
_BASIS_INDICES = onp.arange(1 << NUM_QUBITS, dtype=onp.intp)
CNOT_PERMUTATIONS = tuple(
    _BASIS_INDICES ^ (
        ((_BASIS_INDICES >> (NUM_QUBITS - 1 - control)) & 1)
        << (NUM_QUBITS - 2 - control)
    )
    for control in range(NUM_QUBITS - 1)
)


def hf_aligned_center():

    preimage = HF_STATE.copy()
    for _ in range(NUM_LAYERS):
        for wire in range(NUM_QUBITS - 2, -1, -1):
            preimage[wire + 1] ^= preimage[wire]

    weights = onp.zeros(PARAM_SHAPE, dtype=float)
    weights[0, :, 0] = onp.pi * (preimage != HF_STATE)
    return weights.reshape(-1)


HF_ALIGNED_CENTER = hf_aligned_center()
OCCUPATION_COUNTS = onp.fromiter(
    (bin(basis_index).count("1") for basis_index in range(1 << NUM_QUBITS)),
    dtype=int,
    count=(1 << NUM_QUBITS),
)


device = qml.device("default.qubit", wires=NUM_QUBITS + 1)


def ansatz(params):
    weights = qml.math.reshape(params, PARAM_SHAPE)
    qml.BasisState(HF_STATE, wires=range(NUM_QUBITS))

    for layer in range(PARAM_SHAPE[0]):
        for wire in range(NUM_QUBITS):
            qml.RY(weights[layer, wire, 0], wires=wire)
            qml.RZ(weights[layer, wire, 1], wires=wire)

        for wire in range(NUM_QUBITS - 1):
            qml.CNOT(wires=[wire, wire + 1])


@qml.qnode(device)
def energy_circuit(params):
    ansatz(params)
    return qml.expval(HAMILTONIAN)


@qml.qnode(device)
def single_observable_circuit(params, observable):
    ansatz(params)
    return qml.expval(observable)


@qml.qnode(device)
def state_probabilities_circuit(params):
    ansatz(params)
    return qml.probs(wires=range(NUM_QUBITS))


state_device = qml.device("default.qubit", wires=NUM_QUBITS)


@qml.qnode(state_device)
def reference_state_circuit(params):
    ansatz(params)
    return qml.state()


ENERGY_GRADIENT_FN = qml.grad(energy_circuit, argnum=0)
QNG_METRIC_FN = qml.metric_tensor(energy_circuit, approx="block-diag")
SINGLE_TERM_GRADIENT_FN = qml.grad(single_observable_circuit, argnum=0)


def call_autograd(function, *args):

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Casting complex values to real discards the imaginary part",
            module=r"autograd\.numpy\.numpy_wrapper",
        )
        return function(*args)


def as_numpy(value):
    return onp.asarray(qml.math.toarray(qml.math.detach(value)))


def as_trainable(values):
    return np.array(values, requires_grad=True)


def apply_single_qubit(batch, wire, gate, derivative=None, parameter_index=None):

    batch_size = batch.shape[0]
    permutation, inverse = WIRE_SPECS[wire]
    blocks = batch.reshape((batch_size,) + (2,) * NUM_QUBITS)
    blocks = blocks.transpose(
        (0,) + tuple(index + 1 for index in permutation)
    ).reshape(batch_size, -1, 2)
    updated = onp.empty_like(blocks)
    zero, one = blocks[:, :, 0], blocks[:, :, 1]
    updated[:, :, 0] = gate[0, 0] * zero + gate[0, 1] * one
    updated[:, :, 1] = gate[1, 0] * zero + gate[1, 1] * one
    if parameter_index is not None:
        updated[parameter_index + 1, :, 0] += (
            derivative[0, 0] * zero[0] + derivative[0, 1] * one[0]
        )
        updated[parameter_index + 1, :, 1] += (
            derivative[1, 0] * zero[0] + derivative[1, 1] * one[0]
        )
    batch = updated.reshape((batch_size,) + (2,) * NUM_QUBITS)
    return batch.transpose(
        (0,) + tuple(index + 1 for index in inverse)
    ).reshape(batch_size, -1)


def su2_state_and_jacobian(values, with_jacobian=True):

    angles = onp.asarray(values, dtype=float).reshape(-1)
    if angles.size != NUM_PARAMS:
        raise ValueError(f"Expected {NUM_PARAMS} parameters, got {angles.size}.")
    angles = angles.reshape(PARAM_SHAPE)
    batch_size = NUM_PARAMS + 1 if with_jacobian else 1
    batch = onp.zeros((batch_size, 1 << NUM_QUBITS), dtype=complex)
    batch[0, int("".join(map(str, HF_STATE)), 2)] = 1.0

    for layer in range(NUM_LAYERS):
        for wire in range(NUM_QUBITS):
            ry_angle, rz_angle = angles[layer, wire]
            parameter_index = 2 * (layer * NUM_QUBITS + wire)
            cosine = onp.cos(ry_angle / 2.0)
            sine = onp.sin(ry_angle / 2.0)
            ry = onp.array([[cosine, -sine], [sine, cosine]])
            dry = 0.5 * onp.array([
                [-sine, -cosine], [cosine, -sine]
            ])
            batch = apply_single_qubit(
                batch, wire, ry, dry,
                parameter_index if with_jacobian else None,
            )

            phase_zero = onp.exp(-0.5j * rz_angle)
            phase_one = onp.exp(0.5j * rz_angle)
            rz = onp.array([
                [phase_zero, 0.0], [0.0, phase_one]
            ])
            drz = onp.array([
                [-0.5j * phase_zero, 0.0],
                [0.0, 0.5j * phase_one],
            ])
            batch = apply_single_qubit(
                batch, wire, rz, drz,
                parameter_index + 1 if with_jacobian else None,
            )

        for permutation in CNOT_PERMUTATIONS:
            batch = batch[:, permutation]

    return batch[0], batch[1:] if with_jacobian else None


def evaluate_energy(params):
    state, _ = su2_state_and_jacobian(as_numpy(params), with_jacobian=False)
    return float(onp.real(onp.vdot(state, HAMILTONIAN_MATRIX @ state)))


def fast_energy_gradient(params):
    state, derivatives = su2_state_and_jacobian(as_numpy(params))
    return 2.0 * onp.real(derivatives.conj() @ (HAMILTONIAN_MATRIX @ state))


def particle_sector_probabilities(final_params):
    state, _ = su2_state_and_jacobian(final_params, with_jacobian=False)
    probabilities = onp.abs(state) ** 2
    sectors = onp.bincount(
        OCCUPATION_COUNTS,
        weights=probabilities,
        minlength=NUM_QUBITS + 1,
    )
    if not onp.isclose(sectors.sum(), 1.0, atol=1.0e-8):
        raise RuntimeError("Particle-sector probabilities do not sum to one.")
    return sectors


def initial_parameters(run_index):
    rng = onp.random.default_rng(BASE_SEED + run_index)
    center = HF_ALIGNED_CENTER if INITIALIZATION_CENTER == "hf" else 0.0
    return center + rng.uniform(
        INITIALIZATION_LOW,
        INITIALIZATION_HIGH,
        size=NUM_PARAMS,
    )


def validate_gradient():
    initial = initial_parameters(0)
    fast_state, _ = su2_state_and_jacobian(initial, with_jacobian=False)
    reference_state = onp.asarray(
        as_numpy(reference_state_circuit(as_trainable(initial)))
    )
    overlap = onp.vdot(reference_state, fast_state)
    if abs(overlap) < 1.0e-12:
        raise RuntimeError("Fast SU2 state has zero overlap with PennyLane.")
    aligned_state = fast_state * onp.conj(overlap) / abs(overlap)
    if onp.linalg.norm(reference_state - aligned_state) > 1.0e-8:
        raise RuntimeError("Fast SU2 state disagrees with PennyLane.")
    reference_energy = float(as_numpy(energy_circuit(as_trainable(initial))))
    if abs(evaluate_energy(initial) - reference_energy) > 1.0e-8:
        raise RuntimeError("Fast SU2 energy disagrees with PennyLane.")
    analytic = onp.real(
        as_numpy(call_autograd(ENERGY_GRADIENT_FN, as_trainable(initial)))
    )
    fast_gradient, fast_metric, term_gradients = compute_hqng_quantities(
        initial, return_terms=True
    )
    if not onp.allclose(fast_gradient, analytic, atol=1.0e-6, rtol=1.0e-5):
        raise RuntimeError("Fast H-QNG gradient disagrees with PennyLane.")
    if not onp.allclose(
        fast_energy_gradient(initial), analytic, atol=1.0e-6, rtol=1.0e-5
    ):
        raise RuntimeError("Fast energy gradient disagrees with PennyLane.")
    if not onp.all(onp.isfinite(fast_metric)):
        raise RuntimeError("Fast H-QNG metric contains non-finite values.")
    active_terms = onp.argsort(onp.linalg.norm(term_gradients, axis=1))[-3:]
    for term_index in active_terms:
        reference = onp.real(as_numpy(call_autograd(
            SINGLE_TERM_GRADIENT_FN,
            as_trainable(initial),
            HAMILTONIAN_TERMS[term_index],
        )))
        if not onp.allclose(
            term_gradients[term_index], reference,
            atol=1.0e-6, rtol=1.0e-5,
        ):
            raise RuntimeError(
                f"Fast SU2 term gradient disagrees for term {term_index}."
            )
    indices = onp.unique(
        onp.linspace(0, NUM_PARAMS - 1, num=min(5, NUM_PARAMS), dtype=int)
    )
    numerical = onp.empty(indices.size, dtype=float)

    for position, index in enumerate(indices):
        plus = initial.copy()
        minus = initial.copy()
        plus[index] += GRADIENT_CHECK_EPSILON
        minus[index] -= GRADIENT_CHECK_EPSILON
        numerical[position] = (
            evaluate_energy(as_trainable(plus))
            - evaluate_energy(as_trainable(minus))
        ) / (2.0 * GRADIENT_CHECK_EPSILON)

    absolute_errors = onp.abs(analytic[indices] - numerical)
    allowed_errors = (
        GRADIENT_CHECK_ATOL + GRADIENT_CHECK_RTOL * onp.abs(numerical)
    )
    if not onp.all(absolute_errors <= allowed_errors):
        raise RuntimeError(
            "Energy-gradient validation failed: "
            f"indices={indices.tolist()}, "
            f"analytic={analytic[indices].tolist()}, "
            f"numerical={numerical.tolist()}, "
            f"errors={absolute_errors.tolist()}"
        )
    print(
        "Gradient validation passed: "
        f"{indices.size} parameters; max absolute error="
        f"{absolute_errors.max():.3e}.",
        flush=True,
    )


def compute_hqng_quantities(params, return_terms=False):

    state, derivatives = su2_state_and_jacobian(as_numpy(params))
    acted_states = onp.stack(
        [matrix @ state for matrix in TERM_MATRICES], axis=0
    )
    term_gradients = 2.0 * onp.real(
        derivatives.conj() @ acted_states.T
    ).T
    gradient = HAMILTONIAN_COEFFICIENTS @ term_gradients
    weighted = (HAMILTONIAN_COEFFICIENTS**2)[:, None] * term_gradients
    metric = term_gradients.T @ weighted / HQNG_NORMALIZATION
    if return_terms:
        return gradient, metric, term_gradients
    return gradient, metric


def report_progress(run_index, method, step, energy):
    if step % PROGRESS_EVERY == 0 or step == NUM_STEPS:
        print(
            f"run {run_index:02d} | {method:<5} | "
            f"step {step:03d}/{NUM_STEPS} | E={energy:.10f}",
            flush=True,
        )


def run_vg(initial, run_index):
    params = as_trainable(initial)
    energies = onp.empty(NUM_STEPS + 1, dtype=float)
    energies[0] = evaluate_energy(params)

    for step in range(1, NUM_STEPS + 1):
        gradient = fast_energy_gradient(params)
        params = as_trainable(as_numpy(params) - LEARNING_RATE * gradient)
        energies[step] = evaluate_energy(params)
        report_progress(run_index, "VG", step, energies[step])

    return energies, as_numpy(params)


def run_qng(initial, run_index):
    params = as_trainable(initial)
    energies = onp.empty(NUM_STEPS + 1, dtype=float)
    energies[0] = evaluate_energy(params)

    for step in range(1, NUM_STEPS + 1):
        metric = onp.real(as_numpy(call_autograd(QNG_METRIC_FN, params)))
        gradient = fast_energy_gradient(params)
        direction = onp.linalg.pinv(metric) @ gradient
        params = as_trainable(as_numpy(params) - LEARNING_RATE * direction)
        energies[step] = evaluate_energy(params)
        report_progress(run_index, "QNG", step, energies[step])

    return energies, as_numpy(params)


def run_hqng(initial, run_index):
    params = as_trainable(initial)
    identity = onp.eye(NUM_PARAMS)
    energies = onp.empty(NUM_STEPS + 1, dtype=float)
    energies[0] = evaluate_energy(params)

    for step in range(1, NUM_STEPS + 1):
        gradient, metric = compute_hqng_quantities(params)
        direction = onp.linalg.solve(
            metric + HQNG_REGULARIZATION * identity, gradient
        )
        params = as_trainable(as_numpy(params) - LEARNING_RATE * direction)
        energies[step] = evaluate_energy(params)
        report_progress(run_index, "H-QNG", step, energies[step])

    return energies, as_numpy(params)


def process_run(run_index):
    initial = initial_parameters(run_index)
    onp.save(RESULTS_DIR / f"initial_params{run_index}.npy", initial)
    initial_energy = evaluate_energy(as_trainable(initial))
    initial_sector = particle_sector_probabilities(initial)
    print(
        f"run {run_index:02d} | started | E0={initial_energy:.10f} | "
        f"P0(N={TARGET_PARTICLE_NUMBER})="
        f"{initial_sector[TARGET_PARTICLE_NUMBER]:.6f}",
        flush=True,
    )

    for method, optimizer in (
        ("hqng", run_hqng),
        ("vg", run_vg),
        ("qng", run_qng),
    ):
        print(f"run {run_index:02d} | {method.upper()} | started", flush=True)
        energies, final_params = optimizer(initial, run_index)
        sector_probabilities = particle_sector_probabilities(final_params)

        onp.save(RESULTS_DIR / f"value_{method}{run_index}.npy", energies)
        onp.save(
            RESULTS_DIR / f"final_params_{method}{run_index}.npy",
            final_params,
        )
        onp.save(
            RESULTS_DIR / f"particle_sector_{method}{run_index}.npy",
            sector_probabilities,
        )
        print(
            f"run {run_index:02d} | {method.upper()} | completed | "
            f"P(N={TARGET_PARTICLE_NUMBER})="
            f"{sector_probabilities[TARGET_PARTICLE_NUMBER]:.6f}",
            flush=True,
        )

    print(f"run {run_index:02d} | completed", flush=True)
    return run_index


def main():
    start_time = time.perf_counter()
    print(f"Molecule: {MOLECULE}")
    print(f"Ansatz: EfficientSU2-style, [RY/RZ-CNOT(linear)] x {NUM_LAYERS}")
    print(f"Qubits: {NUM_QUBITS}; parameters: {NUM_PARAMS}")
    print(f"CNOTs per circuit: {NUM_LAYERS * (NUM_QUBITS - 1)}")
    print(f"Target electron number: {TARGET_PARTICLE_NUMBER}")
    print(f"FCI energy (fixed electron number): {FCI_ENERGY}")
    print(f"Hamiltonian terms: {len(HAMILTONIAN_TERMS)}")
    print(f"Learning rate: {LEARNING_RATE}; H-QNG lambda: {HQNG_REGULARIZATION}")
    print(
        f"Initialization: {INITIALIZATION_CENTER} center + independent Uniform"
        f"([{INITIALIZATION_LOW:.6f}, {INITIALIZATION_HIGH:.6f}])"
    )
    center_probabilities = onp.asarray(
        as_numpy(state_probabilities_circuit(as_trainable(HF_ALIGNED_CENTER)))
    )
    hf_basis_index = int("".join(str(bit) for bit in HF_STATE), 2)
    hf_probability = float(center_probabilities[hf_basis_index])
    if not onp.isclose(hf_probability, 1.0, atol=1.0e-10, rtol=0.0):
        raise RuntimeError(
            f"HF-aligned center does not prepare the HF state: P={hf_probability}."
        )
    print(
        f"HF-aligned center: E={evaluate_energy(as_trainable(HF_ALIGNED_CENTER)):.10f}; "
        f"P(HF)={hf_probability:.10f}"
    )
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Runs: {NUM_RUNS}; steps: {NUM_STEPS}; workers: {MAX_WORKERS}")
    validate_gradient()

    if VALIDATE_ONLY:
        print("Validation-only run completed; optimization was not started.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    if NUM_RUNS == 1:
        process_run(0)
    else:
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
        raise RuntimeError(f"{MOLECULE} experiment failed: {failed_runs}")

    elapsed = time.perf_counter() - start_time
    print(f"Total time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
