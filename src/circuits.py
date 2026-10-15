"""Explicit QAOA and warm-start QAOA circuits.

The circuits are written out gate by gate (rather than taken from a library
class) so that every line corresponds to a term in the paper:

    |psi(gamma, beta)> = prod_{l=1}^{p} U_M(beta_l) U_C(gamma_l) |phi_0>

    U_C(gamma) = exp(-i gamma H_C)           H_C diagonal: Z and ZZ terms only
    U_M(beta)  = exp(-i beta H_M)            H_M = -sum_i X_i             (standard)
                                              H_M = sum_i H_M,i^{ws}      (warm start)

The mixer sign is chosen so that |phi_0> is the ground state of H_M in both
cases (|+>^n for -sum X_i), which is the convention of the adiabatic picture:
a linear ramp with positive angles then lowers the energy. Qiskit's QAOAAnsatz
uses +sum X_i, i.e. beta -> -beta; the two agree up to that sign.

For the warm start, H_M,i^{ws} = (2c_i - 1) Z_i - 2 sqrt(c_i (1 - c_i)) X_i has
ground state R_Y(theta_i)|0>, and exp(-i beta H_M,i^{ws}) = R_Y(theta_i) R_Z(-2 beta) R_Y(-theta_i)
(Egger, Marecek and Woerner, 2021); at c_i = 1/2 it reduces to -X_i, so the
warm-start ansatz contains the standard one as the special case eps = 1/2.
tests/test_circuits.py checks these identities numerically.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def normalise_hamiltonian(H: SparsePauliOp):
    """Scale H_C so that its largest |coefficient| is one; returns (H_norm, scale).

    The scale factor makes angle ranges comparable across instances and
    penalty values; expectation values are always computed on the physical
    energy E(x) from bitstrings, never on the normalised operator.
    """
    scale = float(np.max(np.abs(H.coeffs.real)))
    return H / scale, scale


def _z_terms(H: SparsePauliOp):
    """Yield (qubit indices, coefficient) for every Z / ZZ term of a diagonal H."""
    for pauli, coeff in zip(H.paulis, H.coeffs):
        label = pauli.to_label()[::-1]           # index 0 = qubit 0
        if any(ch in "XY" for ch in label):
            raise ValueError("cost Hamiltonian must be diagonal (Z/ZZ terms only)")
        idx = [i for i, ch in enumerate(label) if ch == "Z"]
        if idx:
            yield idx, float(coeff.real)


def build_qaoa_circuit(H_norm: SparsePauliOp, p: int, thetas=None) -> QuantumCircuit:
    """Parameterised QAOA circuit with ParameterVectors gamma[0..p-1], beta[0..p-1].

    thetas = None  -> standard ansatz (|+>^n and X mixer)
    thetas = array -> warm-start ansatz (biased product state and matching mixer)
    """
    n = H_norm.num_qubits
    gammas = ParameterVector("γ", p)
    betas = ParameterVector("β", p)
    qc = QuantumCircuit(n, name="QAOA" if thetas is None else "WS-QAOA")

    # Initial state |phi_0>
    if thetas is None:
        qc.h(range(n))
    else:
        for i, th in enumerate(thetas):
            qc.ry(float(th), i)

    terms = list(_z_terms(H_norm))
    for l in range(p):
        # Cost layer exp(-i gamma H_C): RZ(2 gamma c) for Z_i, RZZ(2 gamma c) for Z_i Z_j
        for idx, c in terms:
            if len(idx) == 1:
                qc.rz(2.0 * gammas[l] * c, idx[0])
            else:
                qc.rzz(2.0 * gammas[l] * c, idx[0], idx[1])
        qc.barrier()
        # Mixer layer: exp(-i beta (-X)) = RX(-2 beta)
        if thetas is None:
            qc.rx(-2.0 * betas[l], range(n))
        else:
            for i, th in enumerate(thetas):
                qc.ry(-float(th), i)
                qc.rz(-2.0 * betas[l], i)
                qc.ry(float(th), i)
        qc.barrier()
    return qc


def pack_parameters(circuit: QuantumCircuit, gammas, betas) -> np.ndarray:
    """Order (gammas, betas) exactly as ``circuit.parameters`` expects.

    Qiskit sorts parameters by name, so beta[...] comes before gamma[...];
    relying on a fixed concatenation order silently swaps the angles.
    """
    values = {}
    for l, g in enumerate(gammas):
        values[f"γ[{l}]"] = float(g)
    for l, b in enumerate(betas):
        values[f"β[{l}]"] = float(b)
    return np.array([values[prm.name] for prm in circuit.parameters])


def unpack_parameters(circuit: QuantumCircuit, vector) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of pack_parameters: a flat vector -> (gammas, betas)."""
    names = [prm.name for prm in circuit.parameters]
    p = len(names) // 2
    g = np.zeros(p)
    b = np.zeros(p)
    for name, v in zip(names, vector):
        kind, l = name[0], int(name[2:-1])
        (g if kind == "γ" else b)[l] = v
    return g, b
