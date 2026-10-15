"""Hybrid optimisation loop, sampling and the metrics used in the paper.

Because the cost Hamiltonian is diagonal, the expectation value <H_C> is
computed from sampled bitstrings as the shot average of E(x). The same
Sampler primitive therefore serves the optimisation loop on the simulator,
the noisy simulator and real hardware, which keeps the three execution
environments strictly comparable.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.optimize import minimize

from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler

from .circuits import pack_parameters, unpack_parameters
from .qubo import QuboModel, _all_bitstrings, decode, is_valid, qubo_energies, qubo_energy

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------
# Backends and primitives
# ----------------------------------------------------------------------------
def ideal_backend(seed: int | None = None) -> AerSimulator:
    return AerSimulator(seed_simulator=seed)


def noise_model_of(fake_backend):
    """Calibrated noise model (gate errors, T1/T2, readout) of a fake IBM backend."""
    from qiskit_aer.noise import NoiseModel
    return NoiseModel.from_backend(fake_backend)


def make_sampler(backend=None, seed: int | None = None, shots: int = 2000, noise_model=None):
    """Return (backend, sampler).

    * ``backend`` None or an AerSimulator -> qiskit-aer SamplerV2 (seedable); a
      ``noise_model`` turns it into the noisy simulator used for E4/E5.
    * any other BackendV2 (a real QPU) -> IBM Runtime Sampler; the client-side
      ``executor_sampler.Sampler`` is preferred and the legacy ``SamplerV2`` is
      the fallback for older qiskit-ibm-runtime.
    """
    if backend is None:
        backend = ideal_backend(seed)
    if isinstance(backend, AerSimulator):
        options = {"backend_options": {}}
        if noise_model is not None:
            options["backend_options"]["noise_model"] = noise_model
        return backend, AerSampler(default_shots=shots, seed=seed, options=options)
    try:
        from qiskit_ibm_runtime.executor_sampler import Sampler
    except ImportError:            # qiskit-ibm-runtime < 0.50
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots
    return backend, sampler


def transpile_for(circuit, backend, seed_transpiler: int = 0):
    """ISA circuit for the backend (level 1 for Aer, level 3 with a seed for devices).

    For the noisy simulator pass the *fake device* here, so that layout,
    routing and basis gates are those of the real chip; the resulting circuit
    is then executed by Aer with the device's noise model.
    """
    level = 1 if isinstance(backend, AerSimulator) else 3
    pm = generate_preset_pass_manager(backend=backend, optimization_level=level,
                                      seed_transpiler=seed_transpiler)
    return pm.run(circuit)


def sample_counts(isa_circuit, params, sampler, shots: int) -> dict:
    """Run one parameter binding and return {bitstring: count}."""
    job = sampler.run([(isa_circuit, np.asarray(params))], shots=shots)
    return job.result()[0].data.meas.get_counts()


# ----------------------------------------------------------------------------
# Energies from counts
# ----------------------------------------------------------------------------
def bits_from_key(key: str) -> list[int]:
    """Qiskit prints qubit 0 on the right; reverse so that x[0] is qubit 0."""
    return [int(b) for b in key[::-1]]


def energy_table(counts: dict, model: QuboModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bit matrix, energies, counts) for the distinct strings in ``counts``."""
    keys = list(counts.keys())
    X = np.array([bits_from_key(k) for k in keys], dtype=np.int8)
    E = qubo_energies(model, X)
    C = np.array([counts[k] for k in keys], dtype=float)
    return X, E, C


def energy_from_counts(counts: dict, model: QuboModel) -> float:
    _, E, C = energy_table(counts, model)
    return float(np.dot(E, C) / C.sum())


def best_bitstring(counts: dict, model: QuboModel):
    X, E, _ = energy_table(counts, model)
    k = int(E.argmin())
    return X[k].astype(int), float(E[k])


_ENERGY_CACHE: dict = {}


def all_energies(model: QuboModel) -> np.ndarray:
    """E(x) for every bit vector, indexed by the integer whose bit i is x_i."""
    key = (model.Q.tobytes(), model.offset)
    if key not in _ENERGY_CACHE:
        parts = [qubo_energies(model, X) for X in _all_bitstrings(model.num_qubits)]
        _ENERGY_CACHE[key] = np.concatenate(parts)
    return _ENERGY_CACHE[key]


def exact_expectation(bound_circuit, model: QuboModel) -> float:
    """Noise-free, shot-free <E> from the statevector (ideal simulator only)."""
    circ = bound_circuit.remove_final_measurements(inplace=False)
    probs = Statevector(circ).probabilities()      # index = integer with qubit 0 as LSB
    return float(np.dot(probs, all_energies(model)))


# ----------------------------------------------------------------------------
# Hybrid loop
# ----------------------------------------------------------------------------
def run_qaoa(model: QuboModel, circuit, gammas0, betas0, sampler, backend,
             shots: int = 2000, maxiter: int = 150, rhobeg: float = 0.25,
             seed_transpiler: int = 0, callback=None) -> dict:
    """COBYLA on the shot-estimated energy; returns parameters, history and the ISA circuit.

    The parameterised circuit is transpiled once; every iteration only binds
    new angles. ``history`` holds the energy of each evaluation and
    ``best_e_seen`` the lowest-energy bitstring observed at any point, which
    is what a practitioner would keep.
    """
    measured = circuit.copy()
    measured.measure_all()
    isa = transpile_for(measured, backend, seed_transpiler)

    x0 = pack_parameters(isa, gammas0, betas0)
    history, best_e_seen, best_x_seen = [], np.inf, None

    def cost(theta):
        nonlocal best_e_seen, best_x_seen
        counts = sample_counts(isa, theta, sampler, shots)
        X, E, C = energy_table(counts, model)
        e = float(np.dot(E, C) / C.sum())
        k = int(E.argmin())
        if E[k] < best_e_seen:
            best_e_seen, best_x_seen = float(E[k]), X[k].astype(int)
        history.append(e)
        if callback:
            callback(len(history), e)
        return e

    t0 = time.time()
    res = minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": rhobeg})
    gammas, betas = unpack_parameters(isa, res.x)
    return {"gammas": gammas, "betas": betas, "params": res.x, "energy": float(res.fun),
            "history": history, "n_evals": len(history), "isa": isa,
            "best_e_seen": best_e_seen, "best_x_seen": best_x_seen,
            "time_s": time.time() - t0}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def optimality_gap(e: float, e_opt: float, e_rand: float) -> float:
    """Normalised gap g = (E - E_opt) / (E_rand - E_opt); 0 is optimal, 1 is chance level."""
    return float((e - e_opt) / max(e_rand - e_opt, 1e-12))


def distribution_metrics(counts: dict, model: QuboModel, e_opt: float, tol: float = 1e-6,
                         c_opt_valid: float | None = None) -> dict:
    """Success probabilities and energies of a measured distribution.

    p_opt        fraction of shots whose bitstring attains the global optimum E_opt
    p_valid      fraction of shots that decode to a valid timetable
    p_valid_opt  fraction of shots that are a valid timetable of minimum conflict cost
                 (needs c_opt_valid; comparable across penalty values, unlike p_opt)
    cost_valid   mean conflict cost of the valid shots (energy of a valid string is its cost)
    """
    X, E, C = energy_table(counts, model)
    total = C.sum()
    valid = np.array([is_valid(decode(x, model)) for x in X])
    out = {"p_opt": float(C[np.abs(E - e_opt) < tol].sum() / total),
           "p_valid": float(C[valid].sum() / total),
           "e_mean": float(np.dot(E, C) / total),
           "e_best": float(E.min()),
           "cost_valid": float(np.dot(E[valid], C[valid]) / C[valid].sum()) if valid.any() else float("nan")}
    if c_opt_valid is not None:
        out["p_valid_opt"] = float(C[valid & (np.abs(E - c_opt_valid) < tol)].sum() / total)
    return out


def iterations_to_convergence(history, scale: float, tol: float = 0.01, window: int = 10) -> int:
    """First evaluation after which the running minimum improves by less than
    ``tol * scale`` during the next ``window`` evaluations (scale = E_rand - E_opt)."""
    h = np.minimum.accumulate(np.asarray(history, dtype=float))
    for k in range(len(h) - window):
        if h[k] - h[k + window] < tol * scale:
            return k + 1
    return len(h)
