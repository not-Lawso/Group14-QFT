"""Noise models N0 / N1 / N2 for AerSimulator."""
from __future__ import annotations

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

# Transpilation target: IBM-style native gate set.
BASIS_GATES = ["rz", "sx", "x", "cx"]

# (single-qubit depolarizing, two-qubit depolarizing, readout flip probability)
NOISE_LEVELS = {
    "N0": None,                    # ideal
    "N1": (0.001, 0.005, 0.01),    # low
    "N2": (0.01, 0.02, 0.03),      # moderate
}


def build_noise_model(p1: float, p2: float, p_ro: float) -> NoiseModel:
    """Depolarizing gate errors + symmetric readout error.

    * p1 is applied to the physical single-qubit pulses ``sx`` and ``x``.
      ``rz`` is implemented virtually (frame change) on superconducting
      hardware and is therefore treated as error-free.
    * p2 is applied to every ``cx``.
    * Readout: P(read 1 | 0) = P(read 0 | 1) = p_ro on every qubit.
    """
    nm = NoiseModel(basis_gates=BASIS_GATES)
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]))
    return nm


def get_noise_model(level: str) -> NoiseModel | None:
    params = NOISE_LEVELS[level]
    return None if params is None else build_noise_model(*params)


def get_simulator(level: str, seed: int, method: str = "automatic") -> AerSimulator:
    """AerSimulator with the requested noise level attached."""
    nm = get_noise_model(level)
    if nm is None:
        return AerSimulator(method=method, seed_simulator=seed)
    return AerSimulator(method=method, noise_model=nm, seed_simulator=seed)
