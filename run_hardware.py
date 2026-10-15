"""E4/E5: evaluate simulator-optimised parameters under noise, with and without mitigation.

The optimisation loop is never run on hardware. For each requested depth the
parameters of the best simulator run (results/<tag>_params.json, written by
run_experiments.py) are evaluated once in each environment:

    ideal   Aer statevector sampling
    noisy   Aer with the calibrated noise model of a fake IBM backend (default FakeFez,
            a Heron r2 device on the Open Plan); the circuit is transpiled for that device
    real    an IBM Quantum QPU through IBM Quantum Compute (--real)

Per environment and depth the script records
    * raw counts (4,000 shots) -> p_valid, p_opt, mean energy and optimality gap
    * the same counts after M3-style readout mitigation (src/mitigation.py)
    * on a real QPU only: Estimator energies at resilience_level 0 and 1 (TREX)

Budget on the Open Plan: 2 depths x (1 sampler job + 1 calibration job
+ 2 estimator jobs) = 8 jobs, roughly three minutes of QPU time in total.

    python run_hardware.py --tag pilot --courses 6 --reps 1,2            # noisy simulator only
    python run_hardware.py --tag pilot --courses 6 --reps 1,2 --real     # also a real QPU
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from src.circuits import build_qaoa_circuit, normalise_hamiltonian, pack_parameters
from src.mitigation import ReadoutMitigator
from src.qaoa_core import (distribution_metrics, make_sampler, noise_model_of, optimality_gap,
                           sample_counts, transpile_for)
from src.qubo import (build_qubo, brute_force_optimum, load_conflict_graph, penalty_from_lambda,
                      qubo_to_ising, random_energy_mean, sub_instance)


def estimator_energies(qpu, isa_measured, params, H, constant, shots):
    """<E> from the IBM Estimator at resilience levels 0 and 1 (level 1 = TREX)."""
    try:
        from qiskit_ibm_runtime.executor_estimator import Estimator
    except ImportError:
        from qiskit_ibm_runtime import EstimatorV2 as Estimator
    circ = isa_measured.remove_final_measurements(inplace=False)
    obs = H.apply_layout(circ.layout)
    out = {}
    for level in (0, 1):
        est = Estimator(mode=qpu)
        est.options.default_shots = shots
        est.options.resilience_level = level
        t0 = time.time()
        ev = float(est.run([(circ, obs, np.asarray(params))]).result()[0].data.evs)
        out[f"estimator_r{level}"] = ev + constant
        out[f"estimator_r{level}_time_s"] = time.time() - t0
    return out


def evaluate(label, transpile_device, sampler, circuit_measured, params, model, H, constant,
             e_opt, e_rand, shots, qpu=None):
    """Sample once, mitigate, and (on a QPU) query the Estimator."""
    isa = transpile_for(circuit_measured, transpile_device, seed_transpiler=0)
    t0 = time.time()
    counts = sample_counts(isa, params, sampler, shots)
    t_sample = time.time() - t0
    raw = distribution_metrics(counts, model, e_opt)
    mitigator = ReadoutMitigator().calibrate(isa, transpile_device, sampler, shots=shots)
    mitigated = distribution_metrics(mitigator.apply(counts), model, e_opt)
    two_q = sum(v for g, v in isa.count_ops().items() if g in ("cz", "ecr", "cx", "rzz"))
    row = {"backend": label, "qubits": model.num_qubits, "shots": shots,
           "two_qubit_gates": int(two_q), "depth": isa.depth(),
           "raw_e_mean": raw["e_mean"], "raw_gap": optimality_gap(raw["e_mean"], e_opt, e_rand),
           "raw_p_valid": raw["p_valid"], "raw_p_opt": raw["p_opt"],
           "mit_e_mean": mitigated["e_mean"], "mit_gap": optimality_gap(mitigated["e_mean"], e_opt, e_rand),
           "mit_p_valid": mitigated["p_valid"], "mit_p_opt": mitigated["p_opt"],
           "readout_p1_given_0": [float(v) for v in mitigator.p1_given_0],
           "readout_p0_given_1": [float(v) for v in mitigator.p0_given_1],
           "sample_time_s": t_sample}
    if qpu is not None:
        row.update(estimator_energies(qpu, isa, params, H, constant, shots))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/exams.csv")
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--courses", type=int, default=6)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--reps", default="1,2")
    ap.add_argument("--method", default="ws_adaptive")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--fake", default="FakeFez", help="fake backend class used as the noisy simulator")
    ap.add_argument("--skip-noisy", action="store_true")
    ap.add_argument("--real", action="store_true", help="also submit to the least busy real QPU")
    ap.add_argument("--out", default=None, help="output JSON (default results/<tag>_hardware.json)")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    out_path = args.out or f"results/{args.tag}_hardware.json"

    G = sub_instance(load_conflict_graph(args.csv), args.courses)
    model = build_qubo(G, args.slots, penalty_from_lambda(G, args.lam))
    H, constant = qubo_to_ising(model)
    H_norm, _ = normalise_hamiltonian(H)
    _, e_opt, _ = brute_force_optimum(model)
    e_rand = random_energy_mean(model)
    params_all = json.load(open(f"results/{args.tag}_params.json"))

    envs = [("ideal", None)]
    if not args.skip_noisy:
        import qiskit_ibm_runtime.fake_provider as fp
        envs.append((f"noisy:{args.fake}", getattr(fp, args.fake)()))
    if args.real:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        qpu = service.least_busy(operational=True, simulator=False, min_num_qubits=model.num_qubits)
        print("selected QPU:", qpu.name)
        envs.append((f"real:{qpu.name}", qpu))

    rows = []
    for p in [int(v) for v in args.reps.split(",")]:
        key = f"{args.courses}c_p{p}_lam{args.lam:g}_{args.method}"
        pr = params_all[key]
        thetas = None if pr["thetas"] is None else np.array(pr["thetas"])
        circuit = build_qaoa_circuit(H_norm, p, thetas=thetas)
        circuit.measure_all()
        params = pack_parameters(circuit, pr["gammas"], pr["betas"])
        for label, device in envs:
            if device is None:
                backend, sampler = make_sampler(None, seed=0, shots=args.shots)
                row = evaluate(label, backend, sampler, circuit, params, model, H, constant,
                               e_opt, e_rand, args.shots)
            elif label.startswith("noisy"):
                _, sampler = make_sampler(None, seed=0, shots=args.shots, noise_model=noise_model_of(device))
                row = evaluate(label, device, sampler, circuit, params, model, H, constant,
                               e_opt, e_rand, args.shots)
            else:
                _, sampler = make_sampler(device, seed=0, shots=args.shots)
                row = evaluate(label, device, sampler, circuit, params, model, H, constant,
                               e_opt, e_rand, args.shots, qpu=device)
            row.update({"p": p, "method": args.method, "courses": args.courses, "lam": args.lam,
                        "e_opt": e_opt, "e_rand": e_rand, "e_sim_exact": pr["e_final_exact"]})
            rows.append(row)
            print(f"[{label}] p={p} 2q={row['two_qubit_gates']} depth={row['depth']} | raw: "
                  f"gap={row['raw_gap']:.3f} p_valid={row['raw_p_valid']:.3f} p_opt={row['raw_p_opt']:.4f} | "
                  f"mitigated: gap={row['mit_gap']:.3f} p_valid={row['mit_p_valid']:.3f} "
                  f"p_opt={row['mit_p_opt']:.4f} | {row['sample_time_s']:.0f}s")
            with open(out_path, "w") as f:
                json.dump(rows, f, indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    main()
