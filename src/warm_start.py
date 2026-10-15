"""Initialisation strategies for QAOA, including the quality-adaptive warm start
that is the main algorithmic contribution (C2).

Five initialisation arms are compared in the experiments:

    random      standard QAOA, angles drawn uniformly at random
    ramp        standard QAOA, Trotterised-annealing linear ramp (Sack & Serbyn, 2021)
    scaled      standard QAOA, ramp angles shrunk by (1 - q/2)  [parameter-only warm start]
    ws_fixed    warm-start QAOA (Egger et al., 2021) with a fixed regularisation eps = 0.25
    ws_adaptive warm-start QAOA whose regularisation eps(q) adapts to the greedy quality q

The greedy timetable x_0 enters the circuit through the initial product state
|phi_0> = prod_i R_Y(theta_i)|0>, theta_i = 2 arcsin(sqrt(c_i)), with c_i = 1 - eps
if x_0,i = 1 and c_i = eps otherwise, and through the matching mixer of Egger et al.
At eps = 0.5 the ansatz reduces to standard QAOA, so eps(q) interpolates
continuously between "no trust in the greedy solution" and "strong trust".
"""
from __future__ import annotations

import numpy as np

from .classical import greedy_schedule
from .qubo import QuboModel, conflict_weight, encode, total_weight

EPS_MIN, EPS_MAX = 0.10, 0.50       # range of the adaptive regularisation
EPS_FIXED = 0.25                    # default of Egger et al. (2021)
RAMP_DT = 0.75                      # Trotter step of the linear ramp (normalised H_C)

METHODS = ("random", "ramp", "scaled", "ws_fixed", "ws_adaptive")


def solution_quality(G, schedule: dict, num_slots: int) -> float:
    """Quality score q in [0, 1] of a valid timetable.

    q = 1 - C(x_0) / C_rand, where C_rand = W / k is the expected conflict
    weight of a uniformly random valid assignment (each of the W shared-student
    pairs clashes with probability 1/k). q = 1 means conflict-free; q = 0
    means no better than chance. Unlike an energy-based score this cannot
    saturate on every instance, because it never involves the penalty P.
    """
    W = total_weight(G)
    if W == 0:
        return 1.0
    c_rand = W / num_slots
    return float(np.clip(1.0 - conflict_weight(G, schedule) / c_rand, 0.0, 1.0))


def epsilon_from_quality(q: float, eps_min: float = EPS_MIN, eps_max: float = EPS_MAX) -> float:
    """Adaptive regularisation: eps(q) = eps_max - (eps_max - eps_min) q."""
    return float(eps_max - (eps_max - eps_min) * q)


def warm_start_thetas(x0: np.ndarray, eps: float) -> np.ndarray:
    """R_Y angles of the biased product state: theta_i = 2 arcsin(sqrt(c_i))."""
    c = np.where(np.asarray(x0) == 1, 1.0 - eps, eps)
    return 2.0 * np.arcsin(np.sqrt(c))


def ramp_angles(p: int, dt: float = RAMP_DT):
    """Linear annealing ramp discretised with the midpoint rule.

    gamma_l = s_l dt, beta_l = (1 - s_l) dt with s_l = (l - 1/2)/p, l = 1..p.
    This is the Trotterised-annealing initialisation of Sack and Serbyn (2021);
    the midpoint rule keeps the mixer active at p = 1 (their end-point rule
    gives beta_1 = 0 there, i.e. no mixing at all).
    """
    s = (np.arange(1, p + 1) - 0.5) / p
    return s * dt, (1.0 - s) * dt


def random_angles(p: int, rng: np.random.Generator):
    return rng.uniform(0.0, np.pi, p), rng.uniform(0.0, np.pi, p)


def initial_point(method: str, p: int, G, model: QuboModel, rng: np.random.Generator):
    """Return a dict with gammas, betas, thetas (None for the standard ansatz), q, eps, x0.

    The random generator is only consumed by the ``random`` arm, so the other
    arms are deterministic for a given instance.
    """
    if method not in METHODS:
        raise ValueError(f"unknown initialisation {method!r}; choose from {METHODS}")
    sched = greedy_schedule(G, model.num_slots)
    x0 = encode(sched, model)
    q = solution_quality(G, sched, model.num_slots)
    out = {"method": method, "q": q, "x0": x0, "eps": None, "thetas": None}

    if method == "random":
        out["gammas"], out["betas"] = random_angles(p, rng)
        return out

    gammas, betas = ramp_angles(p)
    if method == "scaled":
        gammas, betas = gammas * (1.0 - 0.5 * q), betas * (1.0 - 0.5 * q)
    elif method in ("ws_fixed", "ws_adaptive"):
        eps = EPS_FIXED if method == "ws_fixed" else epsilon_from_quality(q)
        out["eps"] = eps
        out["thetas"] = warm_start_thetas(x0, eps)
    out["gammas"], out["betas"] = gammas, betas
    return out
