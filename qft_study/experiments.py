"""Experiment matrix, execution and aggregation."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from qiskit.quantum_info import Operator

from .circuits import build_qft, input_set, prepare_input, rotation_count, signal_vector
from .dft import classical_cost, classical_dft, dft_matrix, reference_distribution
from .metrics import (classical_fidelity, counts_to_probs, logical_counts,
                      operator_fidelities, resource_counts, state_fidelity_pure,
                      total_variation, transpile_qft)
from .noise import NOISE_LEVELS, get_simulator

SHOTS = 1024
SEEDS = [11, 23, 37, 51, 73]

# Unique circuit configurations: (qubits n, approximation parameter m; None = exact).
CIRCUIT_CONFIGS = [(2, None), (3, None), (3, 2), (4, None), (4, 3), (4, 2)]
NOISE_ORDER = ["N0", "N1", "N2"]


def k_max(n: int, m: int | None) -> int:
    return n if m is None else m


def config_label(n: int, m: int | None) -> str:
    return "Exact" if m is None else f"AQFT m={m}"


def experiment_matrix() -> list[dict]:
    """E1-E18: every unique circuit configuration under N0, N1 and N2."""
    rows, idx = [], 1
    for noise in NOISE_ORDER:
        for n, m in CIRCUIT_CONFIGS:
            rows.append({"exp_id": f"E{idx}", "qubits": n, "m": m,
                         "k_max": k_max(n, m), "variant": config_label(n, m),
                         "noise": noise, "shots": SHOTS, "seeds": len(SEEDS)})
            idx += 1
    return rows


# --------------------------------------------------------------------------
def validate_implementation(log=print) -> dict:
    """Check exact QFT == DFT matrix and that our AQFT matches Qiskit's synthesis."""
    from qiskit.synthesis import synth_qft_full
    out = {}
    for n in (2, 3, 4):
        U = Operator(build_qft(n)).data
        F = dft_matrix(2 ** n)
        err_q = float(np.max(np.abs(U - F)))
        v = np.random.default_rng(0).normal(size=2 ** n) + 0j
        err_fft = float(np.max(np.abs(F @ v - classical_dft(v))))
        out[f"n{n}_qft_vs_dft_maxabs"] = err_q
        out[f"n{n}_fft_vs_matrix_maxabs"] = err_fft
        for m in range(1, n):
            ref = Operator(synth_qft_full(n, approximation_degree=n - m))
            same = Operator(build_qft(n, m)).equiv(ref)
            out[f"n{n}_m{m}_equiv_qiskit_degree{n-m}"] = bool(same)
        log(f"  n={n}: max|U_QFT - F_DFT| = {err_q:.2e}, "
            f"max|FFT - F v| = {err_fft:.2e}")
    return out


def resource_analysis(log=print) -> pd.DataFrame:
    """Transpiled resources for every circuit config (+ two sensitivity variants)."""
    rows = []
    for n, m in CIRCUIT_CONFIGS:
        qc = build_qft(n, m)
        exact = build_qft(n)
        base = {"qubits": n, "m": m, "k_max": k_max(n, m), "variant": config_label(n, m),
                "cp_expected": rotation_count(n, m)}
        base.update(logical_counts(qc))
        base.update({f"opt1_{k}": v for k, v in resource_counts(transpile_qft(qc, 1)).items()
                     if k != "ops"})
        base.update({f"opt3_{k}": v for k, v in resource_counts(transpile_qft(qc, 3)).items()
                     if k != "ops"})
        base.update({f"linear_{k}": v for k, v in
                     resource_counts(transpile_qft(qc, 1, "linear")).items() if k != "ops"})
        noswap = transpile_qft(build_qft(n, m, do_swaps=False), 1)
        base.update({f"noswap_{k}": v for k, v in resource_counts(noswap).items() if k != "ops"})
        base.update(operator_fidelities(qc, exact))
        base.update(classical_cost(2 ** n))
        rows.append(base)
        log(f"  {config_label(n, m):>10s} n={n}: CX={base['opt1_two_qubit_gates']:>2d} "
            f"gates={base['opt1_total_gates']:>3d} depth={base['opt1_depth']:>3d} "
            f"F_proc={base['process_fidelity']:.4f}")
    return pd.DataFrame(rows)


def _state_fidelities(n, m, noise, inputs) -> dict:
    """Density-matrix (noise-inclusive, readout-free) fidelity to the DFT output."""
    sim = get_simulator(noise, seed=SEEDS[0], method="density_matrix")
    circs = []
    for fam, par in inputs:
        qc = transpile_qft(prepare_input(n, fam, par).compose(build_qft(n, m)), 1)
        qc.save_density_matrix()          # added after transpilation (simulator instruction)
        circs.append(qc)
    res = sim.run(circs).result()
    out = {}
    for i, (fam, par) in enumerate(inputs):
        rho = np.asarray(res.data(i)["density_matrix"])
        psi = classical_dft(signal_vector(n, fam, par))
        out[(fam, par)] = state_fidelity_pure(rho, psi)
    return out


def run_experiment(cfg: dict, log=print) -> tuple[list[dict], list[dict], dict]:
    """Run one experiment over all inputs and seeds. Returns per-input rows,
    per-seed summary rows and stored count dictionaries."""
    n, m, noise = cfg["qubits"], cfg["m"], cfg["noise"]
    inputs = input_set(n)
    signals = [(f, p) for f, p in inputs if f != "basis"]

    qft_t = transpile_qft(build_qft(n, m), 1)
    res_counts = resource_counts(qft_t)

    sf = _state_fidelities(n, m, noise, inputs)

    meas_circs = []
    for fam, par in signals:
        qc = prepare_input(n, fam, par).compose(build_qft(n, m))
        qc.measure_all()
        meas_circs.append(transpile_qft(qc, 1))
    full_depth = int(np.mean([c.depth() for c in meas_circs]))

    per_input, per_seed, stored = [], [], {}
    for seed in SEEDS:
        sim = get_simulator(noise, seed=seed)
        result = sim.run(meas_circs, shots=SHOTS, seed_simulator=seed).result()
        fids, tvds = [], []
        for i, (fam, par) in enumerate(signals):
            counts = result.get_counts(i)
            p_hat = counts_to_probs(counts, n)
            q = reference_distribution(signal_vector(n, fam, par))
            F, tvd = classical_fidelity(p_hat, q), total_variation(p_hat, q)
            fids.append(F); tvds.append(tvd)
            per_input.append({"exp_id": cfg["exp_id"], "seed": seed, "family": fam,
                              "param": str(par), "dist_fidelity": F, "tvd": tvd,
                              "state_fidelity": sf[(fam, par)]})
            stored[f"{cfg['exp_id']}|{seed}|{fam}|{par}"] = counts
        per_seed.append({"exp_id": cfg["exp_id"], "seed": seed,
                         "dist_fidelity_mean": float(np.mean(fids)),
                         "tone_fidelity_mean": float(np.mean(
                             [f for f, (fa, _) in zip(fids, signals) if fa == "tone"])),
                         "comb_fidelity_mean": float(np.mean(
                             [f for f, (fa, _) in zip(fids, signals) if fa == "comb"])),
                         "tvd_mean": float(np.mean(tvds))})
    summary = {
        "total_gates": res_counts["total_gates"],
        "two_qubit_gates": res_counts["two_qubit_gates"],
        "depth": res_counts["depth"],
        "full_circuit_depth": full_depth,
        "state_fidelity_all": float(np.mean(list(sf.values()))),
        "state_fidelity_basis": float(np.mean([v for (f, _), v in sf.items() if f == "basis"])),
        "state_fidelity_signals": float(np.mean([v for (f, _), v in sf.items() if f != "basis"])),
        "n_inputs_measured": len(signals),
        "n_inputs_state": len(inputs),
    }
    return per_input, per_seed, {"summary": summary, "counts": stored}


def run_all_experiments(log=print):
    cfgs = experiment_matrix()
    all_inputs, all_seeds, counts_store, summaries = [], [], {}, []
    for cfg in cfgs:
        t0 = time.time()
        pi, ps, extra = run_experiment(cfg, log)
        all_inputs += pi; all_seeds += ps; counts_store.update(extra["counts"])
        df = pd.DataFrame(ps)
        row = dict(cfg); row["m"] = "exact" if cfg["m"] is None else cfg["m"]
        row.update(extra["summary"])
        for col in ("dist_fidelity_mean", "tone_fidelity_mean", "comb_fidelity_mean", "tvd_mean"):
            row[col.replace("_mean", "") + "_mean"] = float(df[col].mean())
            row[col.replace("_mean", "") + "_sd"] = float(df[col].std(ddof=1))
        summaries.append(row)
        log(f"  {cfg['exp_id']:>3s} n={cfg['qubits']} {cfg['variant']:<10s} {cfg['noise']}: "
            f"F_dist={row['dist_fidelity_mean']:.4f}±{row['dist_fidelity_sd']:.4f}  "
            f"F_state={row['state_fidelity_all']:.4f}  CX={row['two_qubit_gates']}  "
            f"depth={row['depth']}  ({time.time()-t0:.1f}s)")
    return (pd.DataFrame(summaries), pd.DataFrame(all_seeds),
            pd.DataFrame(all_inputs), counts_store)


def save_counts(counts_store: dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(counts_store, fh)


# --------------------------------------------------------------------------
# Supplementary experiment S1: noise-strength sweep for n = 4.
# Error rates are scaled jointly along the N2 direction:
#   (p1, p2, p_ro) = s * (1%, 2%, 3%),  s = 0 ... 1.5   (s = 1 reproduces N2)
# --------------------------------------------------------------------------
SWEEP_SCALES = [round(0.1 * i, 1) for i in range(16)]


def run_noise_sweep(n: int = 4, variants=(None, 3, 2), log=print) -> pd.DataFrame:
    from qiskit_aer import AerSimulator
    from .noise import build_noise_model
    inputs = input_set(n)
    signals = [(f, p) for f, p in inputs if f != "basis"]
    rows = []
    for m in variants:
        meas = []
        for fam, par in signals:
            qc = prepare_input(n, fam, par).compose(build_qft(n, m))
            qc.measure_all()
            meas.append(transpile_qft(qc, 1))
        dm = []
        for fam, par in inputs:
            qc = transpile_qft(prepare_input(n, fam, par).compose(build_qft(n, m)), 1)
            qc.save_density_matrix()
            dm.append(qc)
        refs = [reference_distribution(signal_vector(n, f, p)) for f, p in signals]
        psis = [classical_dft(signal_vector(n, f, p)) for f, p in inputs]
        for s in SWEEP_SCALES:
            nm = None if s == 0 else build_noise_model(0.01 * s, 0.02 * s, 0.03 * s)
            per_seed = []
            for seed in SEEDS:
                sim = AerSimulator(noise_model=nm, seed_simulator=seed)
                res = sim.run(meas, shots=SHOTS, seed_simulator=seed).result()
                per_seed.append(np.mean([classical_fidelity(counts_to_probs(res.get_counts(i), n), q)
                                         for i, q in enumerate(refs)]))
            sim_dm = AerSimulator(method="density_matrix", noise_model=nm)
            rdm = sim_dm.run(dm).result()
            f_state = np.mean([state_fidelity_pure(np.asarray(rdm.data(i)["density_matrix"]), psi)
                               for i, psi in enumerate(psis)])
            rows.append({"qubits": n, "variant": config_label(n, m), "scale": s,
                         "p1": 0.01 * s, "p2": 0.02 * s, "p_ro": 0.03 * s,
                         "dist_fidelity_mean": float(np.mean(per_seed)),
                         "dist_fidelity_sd": float(np.std(per_seed, ddof=1)),
                         "state_fidelity": float(f_state)})
        log(f"  S1 sweep done for {config_label(n, m)}")
    return pd.DataFrame(rows)
