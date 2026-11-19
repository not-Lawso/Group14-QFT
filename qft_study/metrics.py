"""Resource counting and fidelity metrics."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, average_gate_fidelity, process_fidelity
from qiskit.transpiler import CouplingMap

from .noise import BASIS_GATES

TRANSPILE_SEED = 2026


def transpile_qft(qc: QuantumCircuit, optimization_level: int = 1,
                  topology: str = "all-to-all") -> QuantumCircuit:
    """Single transpilation methodology used for every configuration."""
    coupling = None
    if topology == "linear":
        coupling = CouplingMap.from_line(qc.num_qubits)
    return transpile(qc, basis_gates=BASIS_GATES, coupling_map=coupling,
                     optimization_level=optimization_level,
                     seed_transpiler=TRANSPILE_SEED)


def resource_counts(qc: QuantumCircuit) -> dict:
    """Gate totals, two-qubit gate count and depth of a (transpiled) circuit."""
    ops = {k: v for k, v in qc.count_ops().items() if k not in ("barrier", "measure")}
    two_q = sum(1 for inst in qc.data
                if inst.operation.num_qubits == 2 and inst.operation.name != "barrier")
    return {
        "total_gates": int(sum(ops.values())),
        "two_qubit_gates": int(two_q),
        "depth": int(qc.depth(lambda inst: inst.operation.name not in ("barrier", "measure"))),
        "ops": dict(ops),
    }


def logical_counts(qc: QuantumCircuit) -> dict:
    """Counts of H / CP / SWAP before transpilation."""
    ops = qc.count_ops()
    return {"h": int(ops.get("h", 0)), "cp": int(ops.get("cp", 0)),
            "swap": int(ops.get("swap", 0)), "logical_depth": int(qc.depth())}


def classical_fidelity(p: np.ndarray, q: np.ndarray) -> float:
    """Bhattacharyya (classical) fidelity F = (sum_k sqrt(p_k q_k))^2."""
    return float(np.sum(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None))) ** 2)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))


def counts_to_probs(counts: dict, n: int) -> np.ndarray:
    """Qiskit count dictionary -> probability vector indexed by integer outcome."""
    p = np.zeros(2 ** n)
    for bitstring, c in counts.items():
        p[int(bitstring.replace(" ", ""), 2)] += c
    return p / p.sum()


def state_fidelity_pure(rho: np.ndarray, psi: np.ndarray) -> float:
    """F = <psi| rho |psi> for a pure reference state."""
    return float(np.real(np.conj(psi) @ rho @ psi))


def operator_fidelities(approx: QuantumCircuit, exact: QuantumCircuit) -> dict:
    """Process and average gate fidelity of an approximate circuit vs the exact QFT."""
    Ua, Ue = Operator(approx), Operator(exact)
    return {"process_fidelity": float(process_fidelity(Ua, Ue)),
            "avg_gate_fidelity": float(average_gate_fidelity(Ua, Ue))}
