"""QUBO formulation of exam timetabling as weighted graph colouring.

Decision variable x_{c,t} = 1 if course c is examined in slot t, else 0.

    E(x) = sum_{(a,b) in E} w_ab sum_t x_{a,t} x_{b,t}            (conflict cost)
         + P * sum_c ( sum_t x_{c,t} - 1 )^2                      (one-slot penalty)

For a valid assignment (exactly one slot per course) the penalty vanishes and
E(x) is the total number of students with two exams in the same slot. The
constant part of the penalty, n_courses * P, is kept in `QuboModel.offset` so
that energies are never negative and are directly interpretable.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_conflict_graph(csv_path: str) -> nx.Graph:
    """Read ``course_a,course_b,shared_students`` and build a weighted conflict graph.

    Duplicate pairs are summed, self-loops are ignored and rows with a
    non-positive number of shared students are dropped.
    """
    df = pd.read_csv(csv_path)
    required = {"course_a", "course_b", "shared_students"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns {sorted(missing)}")
    G = nx.Graph()
    for _, row in df.iterrows():
        a, b = str(row["course_a"]).strip(), str(row["course_b"]).strip()
        w = int(row["shared_students"])
        if a == b or w <= 0:
            continue
        if G.has_edge(a, b):
            G[a][b]["weight"] += w
        else:
            G.add_edge(a, b, weight=w)
    return G


def weighted_degree(G: nx.Graph, course) -> int:
    return int(sum(d["weight"] for _, _, d in G.edges(course, data=True)))


def max_weighted_degree(G: nx.Graph) -> int:
    """Largest total conflict weight incident to a single course, d_max."""
    return max(weighted_degree(G, c) for c in G.nodes())


def total_weight(G: nx.Graph) -> int:
    return int(sum(d["weight"] for _, _, d in G.edges(data=True)))


def sub_instance(G: nx.Graph, n_courses: int) -> nx.Graph:
    """Induced subgraph on the ``n_courses`` courses with the largest weighted degree.

    Ties are broken alphabetically so that the same sub-instance is obtained on
    every machine.
    """
    order = sorted(G.nodes(), key=lambda c: (-weighted_degree(G, c), c))
    return G.subgraph(order[:n_courses]).copy()


# ----------------------------------------------------------------------------
# QUBO model
# ----------------------------------------------------------------------------
@dataclass
class QuboModel:
    """Upper-triangular QUBO matrix plus the bookkeeping needed to decode it."""

    Q: np.ndarray            # upper-triangular, E(x) = x^T Q x + offset
    offset: float            # n_courses * penalty
    courses: list            # sorted course codes
    num_slots: int
    penalty: float
    idx: dict                # (course, slot) -> qubit index

    @property
    def num_qubits(self) -> int:
        return self.Q.shape[0]

    def bit(self, course, slot) -> int:
        return self.idx[(course, slot)]


def penalty_from_lambda(G: nx.Graph, lam: float) -> float:
    """Penalty P = lambda * d_max.

    A course that is left without a slot saves at most d_max conflict weight
    and pays P, so any P > d_max (lambda > 1) guarantees that every optimum is
    a valid timetable. lambda = 1 can tie and lambda < 1 can fail; both are
    exercised deliberately in experiment E6.
    """
    return float(lam * max_weighted_degree(G))


def build_qubo(G: nx.Graph, num_slots: int, penalty: float) -> QuboModel:
    courses = sorted(G.nodes())
    n = len(courses) * num_slots
    idx = {(c, t): i * num_slots + t for i, c in enumerate(courses) for t in range(num_slots)}
    Q = np.zeros((n, n))

    # Conflict term: w_ab * x[a,t] * x[b,t] for every slot t.
    for a, b, data in G.edges(data=True):
        w = float(data["weight"])
        for t in range(num_slots):
            i, j = idx[(a, t)], idx[(b, t)]
            Q[min(i, j), max(i, j)] += w

    # One-slot penalty: P (sum_t x - 1)^2 = P ( -sum_t x + 2 sum_{t<t'} x x' + 1 ),
    # using x^2 = x. The constant P per course goes into the offset.
    for c in courses:
        bits = [idx[(c, t)] for t in range(num_slots)]
        for i in bits:
            Q[i, i] -= penalty
        for i, j in combinations(bits, 2):
            Q[i, j] += 2.0 * penalty

    return QuboModel(Q=Q, offset=float(penalty * len(courses)), courses=courses,
                     num_slots=num_slots, penalty=float(penalty), idx=idx)


def qubo_energy(model: QuboModel, x) -> float:
    """E(x) = x^T Q x + offset for a single bit vector."""
    x = np.asarray(x, dtype=float)
    return float(x @ model.Q @ x + model.offset)


def qubo_energies(model: QuboModel, X: np.ndarray) -> np.ndarray:
    """Vectorised energies for a (m, n) array of bit vectors."""
    X = np.asarray(X, dtype=float)
    return np.einsum("mi,ij,mj->m", X, model.Q, X) + model.offset


def random_energy_mean(model: QuboModel) -> float:
    """Exact mean of E(x) over uniformly random bit vectors (E[x_i]=1/2, E[x_i x_j]=1/4)."""
    Q = model.Q
    diag = np.trace(Q)
    off = Q.sum() - diag
    return float(diag / 2.0 + off / 4.0 + model.offset)


# ----------------------------------------------------------------------------
# Decoding and validation
# ----------------------------------------------------------------------------
def decode(x, model: QuboModel) -> dict:
    """Bit vector -> {course: slot}; a course with zero or several slots maps to None."""
    schedule = {}
    for i, c in enumerate(model.courses):
        slots = [t for t in range(model.num_slots) if x[i * model.num_slots + t] == 1]
        schedule[c] = slots[0] if len(slots) == 1 else None
    return schedule


def encode(schedule: dict, model: QuboModel) -> np.ndarray:
    """{course: slot} -> bit vector."""
    x = np.zeros(model.num_qubits, dtype=int)
    for c, t in schedule.items():
        x[model.idx[(c, t)]] = 1
    return x


def is_valid(schedule: dict) -> bool:
    return all(s is not None for s in schedule.values())


def conflict_weight(G: nx.Graph, schedule: dict) -> int:
    """Students with two exams in the same slot (only meaningful for valid schedules)."""
    return int(sum(d["weight"] for a, b, d in G.edges(data=True)
                   if schedule[a] is not None and schedule[a] == schedule[b]))


def conflict_count(G: nx.Graph, schedule: dict) -> int:
    return int(sum(1 for a, b in G.edges()
                   if schedule[a] is not None and schedule[a] == schedule[b]))


# ----------------------------------------------------------------------------
# Exact references (small instances only)
# ----------------------------------------------------------------------------
def _all_bitstrings(n: int, chunk_bits: int = 16):
    """Yield (m, n) arrays covering every bit vector of length n, chunk by chunk."""
    lo = min(n, chunk_bits)
    hi = n - lo
    low = ((np.arange(2 ** lo)[:, None] >> np.arange(lo)) & 1).astype(np.int8)
    for h in range(2 ** hi):
        high = ((h >> np.arange(hi)) & 1).astype(np.int8)
        yield np.hstack([low, np.tile(high, (low.shape[0], 1))]) if hi else low


def brute_force_optimum(model: QuboModel, max_qubits: int = 24):
    """Global minimum of E(x) over all 2^n bit vectors (valid or not).

    Returns (x_opt, e_opt, n_opt) where n_opt counts degenerate minima.
    Exhaustive enumeration is used, so this is limited to ``max_qubits``.
    """
    n = model.num_qubits
    if n > max_qubits:
        raise ValueError(f"{n} qubits is too many for brute force (limit {max_qubits})")
    best_e, best_x, n_best = np.inf, None, 0
    for X in _all_bitstrings(n):
        E = qubo_energies(model, X)
        m = E.min()
        if m < best_e - 1e-9:
            best_e, best_x, n_best = float(m), X[int(E.argmin())].copy(), int(np.sum(np.isclose(E, m)))
        elif np.isclose(m, best_e):
            n_best += int(np.sum(np.isclose(E, m)))
    return best_x.astype(int), best_e, n_best


def optimum_valid_schedule(G: nx.Graph, model: QuboModel):
    """Minimum-conflict *valid* timetable by enumerating the k^n assignments."""
    from itertools import product
    best_c, best_s, n_best = np.inf, None, 0
    for assign in product(range(model.num_slots), repeat=len(model.courses)):
        s = dict(zip(model.courses, assign))
        c = conflict_weight(G, s)
        if c < best_c:
            best_c, best_s, n_best = c, s, 1
        elif c == best_c:
            n_best += 1
    return best_s, int(best_c), n_best


# ----------------------------------------------------------------------------
# Ising mapping (explicit, for teaching; cross-checked in tests/test_qubo.py)
# ----------------------------------------------------------------------------
def qubo_to_ising(model: QuboModel):
    """Map x_i = (1 - Z_i)/2 and return (H, constant) with H a SparsePauliOp.

    For every computational basis state |x>,  <x|H|x> + constant == E(x).
    Qiskit orders Pauli labels with qubit 0 on the right, hence ``n - 1 - i``.
    """
    from qiskit.quantum_info import SparsePauliOp

    Q = model.Q
    n = Q.shape[0]
    linear = np.zeros(n)          # coefficient of Z_i before the sign flip
    constant = model.offset
    terms = []
    for i in range(n):
        constant += Q[i, i] / 2.0          # Q_ii x_i = Q_ii/2 - Q_ii/2 Z_i
        linear[i] += Q[i, i] / 2.0
        for j in range(i + 1, n):
            c = Q[i, j]
            if abs(c) < 1e-12:
                continue
            # c x_i x_j = c/4 (1 - Z_i - Z_j + Z_i Z_j)
            constant += c / 4.0
            linear[i] += c / 4.0
            linear[j] += c / 4.0
            label = ["I"] * n
            label[n - 1 - i] = "Z"
            label[n - 1 - j] = "Z"
            terms.append(("".join(label), c / 4.0))
    for i in range(n):
        if abs(linear[i]) > 1e-12:
            label = ["I"] * n
            label[n - 1 - i] = "Z"
            terms.append(("".join(label), -linear[i]))
    return SparsePauliOp.from_list(terms).simplify(), float(constant)
