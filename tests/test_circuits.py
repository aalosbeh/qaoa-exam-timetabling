import numpy as np
import pytest
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm

from src.circuits import build_qaoa_circuit, normalise_hamiltonian, pack_parameters, unpack_parameters
from src.qubo import build_qubo, load_conflict_graph, qubo_to_ising, sub_instance
from src.warm_start import warm_start_thetas


def small_hamiltonian():
    G = sub_instance(load_conflict_graph("data/exams.csv"), 4)
    model = build_qubo(G, 2, penalty=20.0)          # 8 qubits keeps the matrices small
    H, _ = qubo_to_ising(model)
    return model, normalise_hamiltonian(H)[0]


def test_standard_circuit_matches_qaoa_ansatz():
    _, H = small_hamiltonian()
    p = 2
    gammas, betas = np.array([0.4, 0.7]), np.array([0.2, 0.9])
    qc = build_qaoa_circuit(H, p)
    ours = qc.assign_parameters(pack_parameters(qc, gammas, betas))
    ref = QAOAAnsatz(cost_operator=H, reps=p)
    # qiskit orders beta before gamma and uses the mixer +sum X, hence -beta
    ref = ref.assign_parameters(list(-betas) + list(gammas))
    fidelity = abs(Statevector(ours).inner(Statevector(ref))) ** 2
    assert fidelity == pytest.approx(1.0, abs=1e-9)


def test_pack_unpack_round_trip():
    _, H = small_hamiltonian()
    qc = build_qaoa_circuit(H, 3)
    g, b = np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])
    g2, b2 = unpack_parameters(qc, pack_parameters(qc, g, b))
    assert np.allclose(g, g2) and np.allclose(b, b2)


def test_warm_start_initial_state_bias():
    model, H = small_hamiltonian()
    x0 = np.array([1, 0, 0, 1, 1, 0, 0, 1])
    eps = 0.2
    qc = build_qaoa_circuit(H, 1, thetas=warm_start_thetas(x0, eps))
    qc = qc.assign_parameters({prm: 0.0 for prm in qc.parameters})  # zero angles: state = |phi_0>
    probs = Statevector(qc).probabilities_dict()
    key = "".join(str(b) for b in x0[::-1])
    assert probs[key] == pytest.approx((1 - eps) ** len(x0))
    # marginal probability of a 1 on qubit i equals c_i
    sv = Statevector(qc)
    for i, bit in enumerate(x0):
        p1 = sum(pr for k, pr in probs.items() if k[::-1][i] == "1")
        assert p1 == pytest.approx(1 - eps if bit else eps)


def test_warm_start_at_half_eps_is_uniform_superposition():
    _, H = small_hamiltonian()
    x0 = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    qc = build_qaoa_circuit(H, 1, thetas=warm_start_thetas(x0, 0.5))
    qc = qc.assign_parameters({prm: 0.0 for prm in qc.parameters})
    probs = Statevector(qc).probabilities()
    assert np.allclose(probs, 1 / len(probs))


def test_warm_start_mixer_equals_exponential_of_ws_hamiltonian():
    from qiskit import QuantumCircuit
    beta, c = 0.37, 0.2
    theta = 2 * np.arcsin(np.sqrt(c))
    qc = QuantumCircuit(1)
    qc.ry(-theta, 0)
    qc.rz(-2 * beta, 0)
    qc.ry(theta, 0)
    U = Operator(qc).data
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    H_ws = (2 * c - 1) * Z - 2 * np.sqrt(c * (1 - c)) * X
    U_ref = expm(-1j * beta * H_ws)
    phase = np.vdot(U_ref.ravel(), U.ravel()) / np.vdot(U_ref.ravel(), U_ref.ravel())
    assert np.allclose(U, phase * U_ref, atol=1e-9)
    # the biased state is the ground state of the mixer
    v = np.array([np.sqrt(1 - c), np.sqrt(c)])
    assert np.allclose(H_ws @ v, -v)
