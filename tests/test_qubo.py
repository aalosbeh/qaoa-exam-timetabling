import itertools

import networkx as nx
import numpy as np
import pytest

from src.qubo import (build_qubo, brute_force_optimum, conflict_weight, decode, encode,
                      is_valid, load_conflict_graph, max_weighted_degree,
                      optimum_valid_schedule, penalty_from_lambda, qubo_energies,
                      qubo_energy, qubo_to_ising, random_energy_mean, sub_instance)


def triangle():
    G = nx.Graph()
    G.add_edge("A", "B", weight=1)
    G.add_edge("B", "C", weight=1)
    G.add_edge("A", "C", weight=1)
    return G


def test_triangle_optimum_is_conflict_free():
    G = triangle()
    model = build_qubo(G, num_slots=3, penalty=10.0)
    x_opt, e_opt, n_opt = brute_force_optimum(model)
    sched = decode(x_opt, model)
    assert is_valid(sched)
    assert conflict_weight(G, sched) == 0
    assert e_opt == pytest.approx(0.0)
    assert n_opt == 6                      # 3! relabellings of the colours


def test_valid_schedule_energy_equals_conflict_weight():
    G = load_conflict_graph("data/exams.csv")
    model = build_qubo(G, 3, penalty=50.0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        sched = {c: int(rng.integers(3)) for c in model.courses}
        assert qubo_energy(model, encode(sched, model)) == pytest.approx(conflict_weight(G, sched))


def test_invalid_schedules_are_penalised():
    model = build_qubo(triangle(), 3, penalty=10.0)
    x = np.zeros(model.num_qubits, dtype=int)           # no course has a slot
    assert qubo_energy(model, x) == pytest.approx(30.0)
    x[0] = x[1] = 1                                      # course A in two slots
    assert qubo_energy(model, x) == pytest.approx(30.0)


def test_ising_mapping_matches_energy_on_basis_states():
    G = load_conflict_graph("data/exams.csv")
    model = build_qubo(sub_instance(G, 4), 3, penalty=20.0)
    H, const = qubo_to_ising(model)
    diag = H.to_matrix(sparse=True).diagonal().real
    for v in np.random.default_rng(1).integers(0, 2 ** model.num_qubits, size=50):
        x = [(int(v) >> i) & 1 for i in range(model.num_qubits)]
        assert diag[v] + const == pytest.approx(qubo_energy(model, x))


def test_random_energy_mean_matches_monte_carlo():
    model = build_qubo(load_conflict_graph("data/exams.csv"), 3, penalty=30.0)
    rng = np.random.default_rng(2)
    X = rng.integers(0, 2, size=(20000, model.num_qubits))
    assert qubo_energies(model, X).mean() == pytest.approx(random_energy_mean(model), rel=0.02)


def test_penalty_bound_guarantees_valid_optimum():
    G = sub_instance(load_conflict_graph("data/exams.csv"), 5)
    for lam in (1.5, 2.0):
        model = build_qubo(G, 3, penalty_from_lambda(G, lam))
        x_opt, e_opt, _ = brute_force_optimum(model)
        assert is_valid(decode(x_opt, model))
        _, c_opt, _ = optimum_valid_schedule(G, model)
        assert e_opt == pytest.approx(c_opt)


def test_small_penalty_can_break_validity():
    G = sub_instance(load_conflict_graph("data/exams.csv"), 4)   # a K4, optimum has one clash
    model = build_qubo(G, 3, penalty=1.0)                         # far below d_max
    x_opt, _, _ = brute_force_optimum(model)
    assert not is_valid(decode(x_opt, model))


def test_sub_instance_is_deterministic_and_nested():
    G = load_conflict_graph("data/exams.csv")
    assert max_weighted_degree(G) == 68
    small = set(sub_instance(G, 4).nodes())
    larger = set(sub_instance(G, 6).nodes())
    assert small <= larger
    assert small == {"MATH240", "MATH201", "CS310", "STAT210"}
