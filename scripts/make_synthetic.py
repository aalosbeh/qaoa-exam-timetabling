"""Generate random weighted conflict graphs with a spread of greedy quality q.

The adaptive warm start can only show its value across instances whose greedy
timetables differ in quality, so experiment E2 is also run on a family of
synthetic instances:

    python scripts/make_synthetic.py --courses 6 --edges 13 --count 20 --out data/synthetic

Each instance is written as data/synthetic/inst_<i>.csv in the same format as
data/exams.csv; a manifest lists its greedy quality and exact optimum.
"""
from __future__ import annotations

import argparse
import itertools
import os
import random
import sys

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.classical import greedy_schedule                      # noqa: E402
from src.qubo import build_qubo, optimum_valid_schedule        # noqa: E402
from src.warm_start import solution_quality                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--courses", type=int, default=6)
    ap.add_argument("--edges", type=int, default=13)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--w-min", type=int, default=3)
    ap.add_argument("--w-max", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/synthetic")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    names = [f"C{i:02d}" for i in range(args.courses)]
    pairs = list(itertools.combinations(names, 2))
    manifest = ["file,courses,edges,q,e_opt,e_greedy"]
    made = 0
    while made < args.count:
        G = nx.Graph()
        G.add_nodes_from(names)
        for a, b in rng.sample(pairs, args.edges):
            G.add_edge(a, b, weight=rng.randint(args.w_min, args.w_max))
        if not nx.is_connected(G):
            continue
        model = build_qubo(G, args.slots, 1.0)
        _, c_opt, _ = optimum_valid_schedule(G, model)
        sched = greedy_schedule(G, args.slots)
        q = solution_quality(G, sched, args.slots)
        e_greedy = sum(d["weight"] for a, b, d in G.edges(data=True) if sched[a] == sched[b])
        path = os.path.join(args.out, f"inst_{made:02d}.csv")
        with open(path, "w") as f:
            f.write("course_a,course_b,shared_students\n")
            for a, b, d in G.edges(data=True):
                f.write(f"{a},{b},{d['weight']}\n")
        manifest.append(f"{os.path.basename(path)},{args.courses},{args.edges},{q:.3f},{c_opt},{e_greedy}")
        made += 1
    with open(os.path.join(args.out, "manifest.csv"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    print("\n".join(manifest))


if __name__ == "__main__":
    main()
