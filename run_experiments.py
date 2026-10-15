"""Experiment grid on the ideal simulator (E1 depth, E2 initialisation, E3 size, E6 penalty).

Every (instance, depth, penalty, seed, initialisation) cell is one hybrid run
with 2,000 shots per energy evaluation and a final 4,000-shot sample. Rows are
appended to results/<tag>_runs.csv as they finish, so an interrupted grid can
be resumed, and the energy history of every run is kept in
results/<tag>_histories.jsonl for the convergence figure. The parameters of
the best run per (instance, depth, method) are saved for run_hardware.py.

Synthetic family (one call per instance file, all rows land in the same CSV):
    for f in data/synthetic/inst_*.csv; do python run_experiments.py --csv $f --sizes 6 --reps 2 --seeds 5 --tag synth; done

Example (smoke test, ~3 minutes):
    python run_experiments.py --sizes 4 --reps 1,2 --lambdas 2 --seeds 3 --tag smoke
Full grid of the proposal (several hours):
    python run_experiments.py --sizes 4,5,6,8 --reps 1,2,3 --lambdas 2 --seeds 20 --tag full
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from src.circuits import build_qaoa_circuit, normalise_hamiltonian
from src.classical import greedy_bits, simulated_annealing, simulated_annealing_native
from src.qaoa_core import (distribution_metrics, exact_expectation, iterations_to_convergence,
                           make_sampler, optimality_gap, run_qaoa, sample_counts)
from src.qubo import (build_qubo, brute_force_optimum, load_conflict_graph, optimum_valid_schedule,
                      penalty_from_lambda, qubo_energy, qubo_to_ising, random_energy_mean, sub_instance)
from src.warm_start import METHODS, initial_point


def parse_list(text, cast):
    return [cast(v) for v in str(text).split(",") if v.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/exams.csv")
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--sizes", default="4,5,6,8", help="numbers of courses (sub-instances by weighted degree)")
    ap.add_argument("--reps", default="1,2,3", help="QAOA depths p")
    ap.add_argument("--lambdas", default="2", help="penalty multipliers, P = lambda * d_max")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--final-shots", type=int, default=4000)
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--tag", default="grid")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    runs_path = f"results/{args.tag}_runs.csv"
    hist_path = f"results/{args.tag}_histories.jsonl"
    params_path = f"results/{args.tag}_params.json"

    done = set()
    if os.path.exists(runs_path):
        prev = pd.read_csv(runs_path)
        changed = False
        if "instance" not in prev.columns:          # logs written before the column existed
            prev.insert(0, "instance", os.path.basename(args.csv))
            changed = True
        for col in ("c_opt_valid", "p_valid_opt", "cost_valid"):
            if col not in prev.columns:
                prev[col] = float("nan")
                changed = True
        if changed:
            prev.to_csv(runs_path, index=False)
        done = set(zip(prev.instance, prev.courses, prev.p, prev.lam, prev.seed, prev.method))
        print(f"resuming: {len(done)} runs already in {runs_path}")

    G_full = load_conflict_graph(args.csv)
    instance = os.path.basename(args.csv)
    sizes = parse_list(args.sizes, int)
    reps_list = parse_list(args.reps, int)
    lambdas = parse_list(args.lambdas, float)
    methods = parse_list(args.methods, str)
    best_params = json.load(open(params_path)) if os.path.exists(params_path) else {}

    for k, lam in itertools.product(sizes, lambdas):
        G = sub_instance(G_full, k)
        model = build_qubo(G, args.slots, penalty_from_lambda(G, lam))
        H, _ = qubo_to_ising(model)
        H_norm, scale = normalise_hamiltonian(H)
        x_opt, e_opt, n_opt = brute_force_optimum(model)
        _, c_opt_valid, _ = optimum_valid_schedule(G, model)
        e_rand = random_energy_mean(model)
        e_greedy = qubo_energy(model, greedy_bits(G, model))
        print(f"\n=== {k} courses, {model.num_qubits} qubits, lambda={lam} (P={model.penalty:.0f}), "
              f"E_opt={e_opt:.0f} (x{n_opt}), E_greedy={e_greedy:.0f}, E_rand={e_rand:.1f}")

        for p, seed in itertools.product(reps_list, range(args.seeds)):
            # Classical baselines share the seed with the quantum runs.
            t0 = time.time()
            _, e_sa_q, _ = simulated_annealing(model, seed=seed)
            t_sa_q = time.time() - t0
            t0 = time.time()
            _, e_sa_n, _ = simulated_annealing_native(G, args.slots, seed=seed)
            t_sa_n = time.time() - t0

            for method in methods:
                key = (instance, k, p, lam, seed, method)
                if key in done:
                    continue
                rng = np.random.default_rng(1000 * seed + 17)
                pt = initial_point(method, p, G, model, rng)
                circuit = build_qaoa_circuit(H_norm, p, thetas=pt["thetas"])
                backend, sampler = make_sampler(None, seed=seed, shots=args.shots)
                out = run_qaoa(model, circuit, pt["gammas"], pt["betas"], sampler, backend,
                               shots=args.shots, maxiter=args.maxiter)
                counts = sample_counts(out["isa"], out["params"], sampler, args.final_shots)
                m = distribution_metrics(counts, model, e_opt, c_opt_valid=c_opt_valid)
                bound = out["isa"].assign_parameters(out["params"])
                e_exact = exact_expectation(bound, model)
                p_improve = float(sum(c for key_, c in counts.items()
                                      if qubo_energy(model, [int(b) for b in key_[::-1]]) < e_greedy)
                                  / sum(counts.values()))
                row = {
                    "instance": instance, "courses": k, "qubits": model.num_qubits, "p": p, "lam": lam, "penalty": model.penalty,
                    "seed": seed, "method": method, "q": pt["q"], "eps": pt["eps"],
                    "e_init": out["history"][0], "e_final_shots": out["energy"], "e_final_exact": e_exact,
                    "gap_final": optimality_gap(e_exact, e_opt, e_rand),
                    "gap_init": optimality_gap(out["history"][0], e_opt, e_rand),
                    "p_opt": m["p_opt"], "p_valid": m["p_valid"], "p_improve": p_improve,
                    "e_best_sampled": m["e_best"], "best_e_seen": out["best_e_seen"],
                    "n_evals": out["n_evals"],
                    "iters_conv": iterations_to_convergence(out["history"], e_rand - e_opt),
                    "time_qaoa_s": out["time_s"], "e_opt": e_opt, "n_opt": n_opt, "e_rand": e_rand,
                    "e_greedy": e_greedy, "e_sa_qubo": e_sa_q, "e_sa_native": e_sa_n,
                    "time_sa_qubo_s": t_sa_q, "time_sa_native_s": t_sa_n,
                    "gammas": json.dumps([float(g) for g in out["gammas"]]),
                    "betas": json.dumps([float(b) for b in out["betas"]]),
                    "c_opt_valid": c_opt_valid, "p_valid_opt": m["p_valid_opt"], "cost_valid": m["cost_valid"],
                }
                pd.DataFrame([row]).to_csv(runs_path, mode="a", header=not os.path.exists(runs_path), index=False)
                with open(hist_path, "a") as f:
                    f.write(json.dumps({"instance": instance, "courses": k, "p": p, "lam": lam, "seed": seed, "method": method,
                                        "history": [float(h) for h in out["history"]]}) + "\n")
                pkey = f"{k}c_p{p}_lam{lam:g}_{method}" if instance == "exams.csv" else f"{instance}_{k}c_p{p}_lam{lam:g}_{method}"
                if pkey not in best_params or e_exact < best_params[pkey]["e_final_exact"]:
                    best_params[pkey] = {"courses": k, "p": p, "lam": lam, "method": method, "seed": seed,
                                         "gammas": [float(g) for g in out["gammas"]],
                                         "betas": [float(b) for b in out["betas"]],
                                         "thetas": None if pt["thetas"] is None else [float(t) for t in pt["thetas"]],
                                         "eps": pt["eps"], "q": pt["q"], "e_final_exact": e_exact}
                    json.dump(best_params, open(params_path, "w"), indent=1)
                print(f"k={k} p={p} lam={lam:g} seed={seed:2d} {method:12s} E0={out['history'][0]:7.1f} "
                      f"E={e_exact:7.1f} gap={row['gap_final']:.3f} p_valid={m['p_valid']:.3f} "
                      f"p_opt={m['p_opt']:.4f} evals={out['n_evals']:3d} conv={row['iters_conv']:3d} "
                      f"{out['time_s']:.1f}s")
    print("done:", runs_path)


if __name__ == "__main__":
    main()
