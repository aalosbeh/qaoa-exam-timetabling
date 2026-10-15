"""One end-to-end run on the ideal simulator, printed as a small report.

    python run_simulator.py --csv data/exams.csv --courses 6 --slots 3 --reps 2 --seed 1

Builds the QUBO, prints the instance summary, runs the greedy timetable, both
Simulated Annealing baselines and QAOA with each of the five initialisations,
and writes results/single_run.json plus the warm-start timetable.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.circuits import build_qaoa_circuit, normalise_hamiltonian
from src.classical import greedy_schedule, simulated_annealing, simulated_annealing_native
from src.qaoa_core import (distribution_metrics, exact_expectation, make_sampler, optimality_gap,
                           run_qaoa, sample_counts)
from src.qubo import (build_qubo, brute_force_optimum, conflict_weight, decode, encode,
                      load_conflict_graph, penalty_from_lambda, qubo_energy, qubo_to_ising,
                      random_energy_mean, sub_instance)
from src.warm_start import METHODS, initial_point, solution_quality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/exams.csv")
    ap.add_argument("--courses", type=int, default=6, help="use the k highest-degree courses (0 = all)")
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--maxiter", type=int, default=150)
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    G = load_conflict_graph(args.csv)
    if args.courses:
        G = sub_instance(G, args.courses)
    model = build_qubo(G, args.slots, penalty_from_lambda(G, args.lam))
    H, _ = qubo_to_ising(model)
    H_norm, scale = normalise_hamiltonian(H)
    e_rand = random_energy_mean(model)
    x_opt, e_opt, n_opt = brute_force_optimum(model)
    print(f"instance: {len(model.courses)} courses, {G.number_of_edges()} conflict pairs, "
          f"{args.slots} slots -> {model.num_qubits} qubits, {len(H)} Pauli terms, P = {model.penalty:.0f}")
    print(f"exact optimum E_opt = {e_opt:.0f} students in clashing pairs ({n_opt} optimal bitstrings), "
          f"E_rand = {e_rand:.1f}")

    greedy = greedy_schedule(G, args.slots)
    e_greedy = conflict_weight(G, greedy)
    q = solution_quality(G, greedy, args.slots)
    print(f"greedy timetable: E = {e_greedy}, quality q = {q:.2f}")
    _, e_sa_q, _ = simulated_annealing(model, seed=args.seed)
    _, e_sa_n, _ = simulated_annealing_native(G, args.slots, seed=args.seed)
    print(f"simulated annealing: QUBO moves E = {e_sa_q:.0f}, native moves E = {e_sa_n}")

    backend, sampler = make_sampler(None, seed=args.seed, shots=args.shots)
    report = {"instance": {"courses": model.courses, "qubits": model.num_qubits, "penalty": model.penalty,
                           "e_opt": e_opt, "e_rand": e_rand, "e_greedy": e_greedy, "q": q,
                           "e_sa_qubo": e_sa_q, "e_sa_native": e_sa_n}, "qaoa": {}}
    best_schedule = None
    print(f"\n{'initialisation':14s} {'E_init':>8s} {'E_final':>8s} {'gap':>6s} {'P(valid)':>9s} {'P(opt)':>7s} {'evals':>6s}")
    for method in METHODS:
        pt = initial_point(method, args.reps, G, model, np.random.default_rng(args.seed))
        circuit = build_qaoa_circuit(H_norm, args.reps, thetas=pt["thetas"])
        out = run_qaoa(model, circuit, pt["gammas"], pt["betas"], sampler, backend,
                       shots=args.shots, maxiter=args.maxiter)
        counts = sample_counts(out["isa"], out["params"], sampler, 4000)
        m = distribution_metrics(counts, model, e_opt)
        e_exact = exact_expectation(out["isa"].assign_parameters(out["params"]), model)
        gap = optimality_gap(e_exact, e_opt, e_rand)
        print(f"{method:14s} {out['history'][0]:8.1f} {e_exact:8.1f} {gap:6.3f} {m['p_valid']:9.3f} "
              f"{m['p_opt']:7.4f} {out['n_evals']:6d}")
        report["qaoa"][method] = {"gammas": out["gammas"].tolist(), "betas": out["betas"].tolist(),
                                  "e_final_exact": e_exact, "gap": gap, "p_valid": m["p_valid"],
                                  "p_opt": m["p_opt"], "n_evals": out["n_evals"], "history": out["history"]}
        if method == "ws_adaptive":
            # most frequent valid, lowest-energy timetable from the final distribution
            best = min(((qubo_energy(model, [int(b) for b in k[::-1]]), -c, k) for k, c in counts.items()))
            best_schedule = decode([int(b) for b in best[2][::-1]], model)

    with open("results/single_run.json", "w") as f:
        json.dump(report, f, indent=1)
    print("\nTimetable from the adaptive warm start (lowest-energy sampled bitstring):")
    for c, t in sorted(best_schedule.items()):
        print(f"  {c:10s} -> slot {t}")
    print(f"  clashing students: {conflict_weight(G, best_schedule) if all(v is not None for v in best_schedule.values()) else 'invalid'}")
    print("saved results/single_run.json")


if __name__ == "__main__":
    main()
