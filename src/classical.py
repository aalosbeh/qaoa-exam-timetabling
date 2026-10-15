"""Classical components: the greedy timetable used for warm-starting and the
Simulated Annealing baseline that is run on the very same QUBO.
"""
from __future__ import annotations

import numpy as np

from .qubo import QuboModel, encode, qubo_energy, weighted_degree


def greedy_schedule(G, num_slots: int) -> dict:
    """Weighted greedy k-colouring (largest weighted degree first).

    Courses are visited in decreasing order of weighted degree (ties broken
    alphabetically) and each is placed in the slot that adds the least
    conflict weight given the courses already placed. The result is always a
    valid timetable; it costs O(n^2 k) and is the classical solution x_0 that
    seeds the warm start.
    """
    order = sorted(G.nodes(), key=lambda c: (-weighted_degree(G, c), c))
    schedule = {}
    for c in order:
        best_slot, best_cost = 0, None
        for t in range(num_slots):
            cost = sum(G[c][b]["weight"] for b in G.neighbors(c) if schedule.get(b) == t)
            if best_cost is None or cost < best_cost:
                best_slot, best_cost = t, cost
        schedule[c] = best_slot
    return schedule


def greedy_bits(G, model: QuboModel) -> np.ndarray:
    return encode(greedy_schedule(G, model.num_slots), model)


def simulated_annealing(model: QuboModel, n_iter: int = 50000, T0: float | None = None,
                        T_end: float = 0.05, seed: int = 0):
    """Single-bit-flip Simulated Annealing on E(x) = x^T Q x + offset.

    Moving a course between slots passes through an invalid state that costs
    the penalty P, so the start temperature defaults to T0 = P and decays
    geometrically to ``T_end`` over ``n_iter`` steps. The energy change of
    flipping bit i is computed in O(n) from the symmetric form of Q, so the
    whole run costs O(n * n_iter). Returns (best_x, best_energy, energy_history).
    """
    rng = np.random.default_rng(seed)
    Q = model.Q
    S = Q + Q.T - np.diag(np.diag(Q))      # symmetric off-diagonal part, diagonal kept once
    n = Q.shape[0]
    x = rng.integers(0, 2, n)
    e = qubo_energy(model, x)
    T = float(model.penalty if T0 is None else T0)
    alpha = (T_end / T) ** (1.0 / n_iter)
    best, best_e = x.copy(), e
    history = [e]
    for _ in range(n_iter):
        i = int(rng.integers(n))
        # Delta for flipping bit i: (1 - 2 x_i) * (Q_ii + sum_{j != i} S_ij x_j)
        delta = (1 - 2 * x[i]) * (Q[i, i] + S[i] @ x - S[i, i] * x[i])
        if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
            x[i] ^= 1
            e += delta
            if e < best_e:
                best, best_e = x.copy(), e
        history.append(e)
        T *= alpha
    return best.astype(int), float(best_e), history


def simulated_annealing_native(G, num_slots: int, n_iter: int = 5000, T0: float | None = None,
                               T_end: float = 0.05, seed: int = 0):
    """Simulated Annealing in the native space of valid timetables.

    A move changes the slot of one course, so every state is a valid
    timetable and the objective is the conflict weight itself. This is the
    practical classical baseline against which the quantum pipeline is
    compared in experiment E3. Returns (best_schedule, best_cost, history).
    """
    rng = np.random.default_rng(seed)
    courses = sorted(G.nodes())
    sched = {c: int(rng.integers(num_slots)) for c in courses}

    def delta(c, new_t):
        old_t = sched[c]
        gain = sum(G[c][b]["weight"] for b in G.neighbors(c) if sched[b] == old_t)
        loss = sum(G[c][b]["weight"] for b in G.neighbors(c) if sched[b] == new_t)
        return loss - gain

    cost = sum(d["weight"] for a, b, d in G.edges(data=True) if sched[a] == sched[b])
    T = float(max(d["weight"] for _, _, d in G.edges(data=True)) if T0 is None else T0)
    alpha = (T_end / T) ** (1.0 / n_iter)
    best, best_c = dict(sched), cost
    history = [cost]
    for _ in range(n_iter):
        c = courses[int(rng.integers(len(courses)))]
        new_t = int((sched[c] + 1 + rng.integers(num_slots - 1)) % num_slots)
        d = delta(c, new_t)
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-12)):
            sched[c] = new_t
            cost += d
            if cost < best_c:
                best, best_c = dict(sched), cost
        history.append(cost)
        T *= alpha
    return best, int(best_c), history
