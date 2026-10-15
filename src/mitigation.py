"""Readout-error mitigation for sampled distributions.

A compact re-implementation of the matrix-free measurement mitigation idea of
Nation et al. (2021, "M3"): per-qubit confusion matrices are estimated from
two calibration circuits (all |0>, all |1>) on exactly the physical qubits
the ISA circuit measures, the assignment matrix is formed only on the
subspace of bitstrings that were actually observed, the linear system is
solved, and the quasi-probabilities are projected onto the nearest proper
probability distribution (Smolin, Gambetta and Smith, 2012).

This mitigates success probabilities and bitstring-based energies; the
expectation-value path on real hardware additionally uses TREX through the
Estimator's ``resilience_level=1``.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

from .qaoa_core import bits_from_key


def _measured_physical_qubits(isa_circuit) -> list[int]:
    """Physical qubit feeding each classical bit of an ISA circuit."""
    if isa_circuit.layout is None:
        return list(range(isa_circuit.num_clbits))
    return list(isa_circuit.layout.final_index_layout())


def calibration_circuits(isa_circuit, backend):
    """All-zeros and all-ones circuits on the same physical qubits as ``isa_circuit``."""
    n = isa_circuit.num_clbits
    phys = _measured_physical_qubits(isa_circuit)
    zeros = QuantumCircuit(n, name="cal_0")
    zeros.measure_all()
    ones = QuantumCircuit(n, name="cal_1")
    ones.x(range(n))
    ones.measure_all()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=0, initial_layout=phys)
    return pm.run([zeros, ones])


class ReadoutMitigator:
    """Per-qubit (tensored) readout model fitted from two calibration circuits."""

    def __init__(self):
        self.p1_given_0 = None    # P(read 1 | prepared 0) per classical bit
        self.p0_given_1 = None    # P(read 0 | prepared 1) per classical bit

    def calibrate(self, isa_circuit, backend, sampler, shots: int = 4000) -> "ReadoutMitigator":
        zeros, ones = calibration_circuits(isa_circuit, backend)
        res = sampler.run([zeros, ones], shots=shots).result()
        c0 = res[0].data.meas.get_counts()
        c1 = res[1].data.meas.get_counts()
        n = isa_circuit.num_clbits
        self.p1_given_0 = _marginal_ones(c0, n)
        self.p0_given_1 = 1.0 - _marginal_ones(c1, n)
        return self

    def assignment_matrix(self, X: np.ndarray) -> np.ndarray:
        """A[i, j] = P(observe string i | true string j) restricted to the observed strings."""
        m, n = X.shape
        A = np.ones((m, m))
        for q in range(n):
            # M_q[a, b] = P(read a | prepared b)
            M = np.array([[1.0 - self.p1_given_0[q], self.p0_given_1[q]],
                          [self.p1_given_0[q], 1.0 - self.p0_given_1[q]]])
            A *= M[np.ix_(X[:, q], X[:, q])]
        return A

    def apply(self, counts: dict) -> dict:
        """Mitigated probability distribution {bitstring: probability} on the observed subspace."""
        keys = list(counts.keys())
        X = np.array([bits_from_key(k) for k in keys], dtype=int)
        p_noisy = np.array([counts[k] for k in keys], dtype=float)
        p_noisy /= p_noisy.sum()
        A = self.assignment_matrix(X)
        quasi = np.linalg.lstsq(A, p_noisy, rcond=None)[0]
        probs = nearest_probability_distribution(quasi)
        return {k: float(p) for k, p in zip(keys, probs) if p > 0}


def _marginal_ones(counts: dict, n: int) -> np.ndarray:
    total = sum(counts.values())
    ones = np.zeros(n)
    for key, c in counts.items():
        ones += c * np.array(bits_from_key(key), dtype=float)
    return ones / total


def nearest_probability_distribution(quasi: np.ndarray) -> np.ndarray:
    """Euclidean projection of a quasi-probability vector onto the probability simplex.

    Sort descending, find the largest rho with u_rho - (sum_{r<=rho} u_r - 1)/rho > 0,
    shift every entry by theta = (sum_{r<=rho} u_r - 1)/rho and clip at zero
    (Smolin, Gambetta and Smith, 2012; Duchi et al., 2008).
    """
    q = np.asarray(quasi, dtype=float)
    u = np.sort(q)[::-1]
    css = np.cumsum(u)
    j = np.arange(1, len(q) + 1)
    rho = np.nonzero(u - (css - 1.0) / j > 0)[0][-1]
    theta = (css[rho] - 1.0) / (rho + 1)
    return np.maximum(q - theta, 0.0)
