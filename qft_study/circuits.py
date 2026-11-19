"""QFT / approximate-QFT circuit construction and test-input preparation.

Conventions
-----------
* Qiskit little-endian ordering: qubit 0 is the least-significant bit, so the
  basis state |x> has integer index x = sum_q b_q 2^q.
* QFT|x> = (1/sqrt(N)) * sum_k exp(+2*pi*i*x*k/N) |k>,  N = 2^n.
* Approximation parameter ``m`` (called k_max in the report): a controlled
  phase rotation R_k = diag(1, exp(2*pi*i/2^k)) is kept only if k <= m.
  For n qubits the exact QFT needs R_2 ... R_n, so any m >= n gives the exact
  QFT; m < n drops the smallest-angle rotations (Coppersmith/Barenco AQFT).
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


def build_qft(n: int, m: int | None = None, do_swaps: bool = True,
              insert_barriers: bool = False) -> QuantumCircuit:
    """Return an exact (m=None or m>=n) or approximate QFT circuit on n qubits."""
    if m is None or m >= n:
        m_eff, label = n, "QFT"
    else:
        if m < 1:
            raise ValueError("m must be >= 1")
        m_eff, label = m, f"AQFT(m={m})"
    qc = QuantumCircuit(n, name=f"{label} n={n}")
    # Process from the most-significant qubit downwards (textbook ordering).
    for target in reversed(range(n)):
        qc.h(target)
        for control in reversed(range(target)):
            k = target - control + 1          # rotation order R_k
            if k <= m_eff:
                qc.cp(2 * np.pi / 2 ** k, control, target)
        if insert_barriers:
            qc.barrier()
    if do_swaps:                              # bit reversal of the output register
        for q in range(n // 2):
            qc.swap(q, n - q - 1)
    return qc


def rotation_count(n: int, m: int | None = None) -> int:
    """Number of controlled-phase gates retained (analytical, used for cross-checks)."""
    m_eff = n if (m is None or m >= n) else m
    return sum(1 for t in range(n) for c in range(t) if t - c + 1 <= m_eff)


# --------------------------------------------------------------------------
# Test inputs.  All three families are product states, so preparing them needs
# only single-qubit gates; any two-qubit error therefore comes from the QFT.
# --------------------------------------------------------------------------
def signal_vector(n: int, family: str, param: tuple) -> np.ndarray:
    """Classical description of the normalised input signal x[0..N-1]."""
    N = 2 ** n
    x = np.arange(N)
    if family == "basis":                      # computational basis state |j>
        (j,) = param
        v = np.zeros(N, dtype=complex); v[j] = 1.0
    elif family == "tone":                     # complex periodic signal, frequency j
        (j,) = param
        v = np.exp(-2j * np.pi * j * x / N) / np.sqrt(N)
    elif family == "comb":                     # impulse train, period r, offset o
        r, o = param
        v = ((x % r) == o).astype(complex)
        v /= np.linalg.norm(v)
    else:
        raise ValueError(family)
    return v


def prepare_input(n: int, family: str, param: tuple) -> QuantumCircuit:
    """Single-qubit-only circuit that prepares ``signal_vector(n, family, param)``."""
    N = 2 ** n
    qc = QuantumCircuit(n, name=f"prep_{family}{param}")
    if family == "basis":
        (j,) = param
        for q in range(n):
            if (j >> q) & 1:
                qc.x(q)
    elif family == "tone":
        (j,) = param
        for q in range(n):
            qc.h(q)
            qc.p(-2 * np.pi * j * (2 ** q) / N, q)
    elif family == "comb":
        r, o = param
        s = int(np.log2(r))
        if 2 ** s != r or not (0 <= o < r):
            raise ValueError("comb period must be a power of two and 0 <= o < r")
        for q in range(s):                     # low bits fixed to the offset
            if (o >> q) & 1:
                qc.x(q)
        for q in range(s, n):                  # high bits in uniform superposition
            qc.h(q)
    return qc


def input_set(n: int) -> list[tuple[str, tuple]]:
    """All test inputs used for an n-qubit experiment."""
    N = 2 ** n
    inputs = [("basis", (j,)) for j in range(N)]
    inputs += [("tone", (j,)) for j in range(N)]
    r = 2
    while r <= N // 2:
        inputs += [("comb", (r, o)) for o in range(r)]
        r *= 2
    return inputs
