import numpy as np
import pytest

from src.classical import greedy_schedule, simulated_annealing
from src.qubo import (build_qubo, brute_force_optimum, conflict_weight, decode, is_valid,
                      load_conflict_graph, optimum_valid_schedule, penalty_from_lambda,
                      qubo_energy, sub_instance)
from src.warm_start import (EPS_FIXED, EPS_MAX, EPS_MIN, METHODS, epsilon_from_quality,
                            initial_point, solution_quality)


@pytest.fixture
def instance():
    G = sub_instance(load_conflict_graph("data/exams.csv"), 6)
    return G, build_qubo(G, 3, penalty_from_lambda(G, 2.0))


def test_greedy_is_valid_and_better_than_chance(instance):
    G, model = instance
    sched = greedy_schedule(G, 3)
    assert is_valid(sched)
    q = solution_quality(G, sched, 3)
    assert 0.0 < q < 1.0
    _, c_opt, _ = optimum_valid_schedule(G, model)
    assert conflict_weight(G, sched) >= c_opt          # greedy is a valid upper bound


def test_quality_is_one_for_conflict_free_schedule():
    import networkx as nx
    G = nx.path_graph(["A", "B", "C"])
    nx.set_edge_attributes(G, 1, "weight")
    assert solution_quality(G, {"A": 0, "B": 1, "C": 0}, 2) == pytest.approx(1.0)


def test_epsilon_endpoints():
    assert epsilon_from_quality(0.0) == pytest.approx(EPS_MAX)
    assert epsilon_from_quality(1.0) == pytest.approx(EPS_MIN)
    assert EPS_MIN < EPS_FIXED < EPS_MAX


def test_all_arms_produce_consistent_initial_points(instance):
    G, model = instance
    rng = np.random.default_rng(0)
    for method in METHODS:
        pt = initial_point(method, 2, G, model, rng)
        assert len(pt["gammas"]) == len(pt["betas"]) == 2
        if method.startswith("ws"):
            assert pt["thetas"].shape == (model.num_qubits,)
            assert 0 < pt["eps"] <= 0.5
        else:
            assert pt["thetas"] is None
    adaptive = initial_point("ws_adaptive", 2, G, model, rng)
    assert adaptive["eps"] == pytest.approx(epsilon_from_quality(adaptive["q"]))


def test_simulated_annealing_reaches_optimum_on_small_instance():
    G = sub_instance(load_conflict_graph("data/exams.csv"), 4)
    model = build_qubo(G, 3, penalty_from_lambda(G, 2.0))
    _, e_opt, _ = brute_force_optimum(model)
    x, e, history = simulated_annealing(model, n_iter=5000, seed=3)
    assert e == pytest.approx(qubo_energy(model, x))
    assert e == pytest.approx(e_opt)
    assert is_valid(decode(x, model))
