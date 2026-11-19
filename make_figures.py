"""Generate all report figures and the statistical comparison table from ./results.

    python make_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import stats

from qft_study.circuits import build_qft, signal_vector
from qft_study.dft import reference_distribution
from qft_study.metrics import resource_counts, transpile_qft

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

# ---- style ---------------------------------------------------------------
if any("Carlito" in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = "Carlito"          # metric-compatible with Calibri
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.linewidth": 0.8,
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})
# Categorical slots (fixed order): exact / k_max=3 / k_max=2
C_EXACT, C_M3, C_M2 = "#2a78d6", "#eb6834", "#1baf7a"
VARIANT_COLOR = {"Exact": C_EXACT, "AQFT m=3": C_M3, "AQFT m=2": C_M2}
# Ordinal noise ramp (single blue hue, light -> dark)
NOISE_COLOR = {"N0": "#86b6ef", "N1": "#2a78d6", "N2": "#104281"}
REF_COLOR = "#0b0b0b"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("  wrote", name)


# ---- circuit diagrams ----------------------------------------------------
def circuit_figures():
    specs = [("fig01_circuit_qft_n2.png", 2, None), ("fig02_circuit_qft_n4.png", 4, None),
             ("fig03_circuit_aqft_n4_m2.png", 4, 2)]
    for fname, n, m in specs:
        qc = build_qft(n, m, insert_barriers=True)
        fig = qc.draw("mpl", fold=-1, style={"name": "bw", "fontsize": 11})
        fig.savefig(os.path.join(FIG, fname), dpi=220, bbox_inches="tight")
        plt.close(fig)
        print("  wrote", fname)


# ---- resource grid: n in {2,3,4} x k_max in {2,3,4} ----------------------
def resource_grid() -> pd.DataFrame:
    rows = []
    for k in (2, 3, 4):
        for n in (2, 3, 4):
            m = None if k >= n else k
            for topo in ("all-to-all", "linear"):
                r = resource_counts(transpile_qft(build_qft(n, m), 1, topo))
                rows.append({"k_max": k, "qubits": n, "topology": topo,
                             "exact": m is None, **{x: r[x] for x in
                                                    ("total_gates", "two_qubit_gates", "depth")}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "resource_grid.csv"), index=False)
    return df


KCOL = {4: C_EXACT, 3: C_M3, 2: C_M2}
KLAB = {4: "k_max = 4 (exact for n ≤ 4)", 3: "k_max = 3", 2: "k_max = 2"}
KMARK = {4: "o", 3: "s", 2: "^"}
KMS = {4: 11, 3: 6, 2: 6}
KLW = {4: 4.5, 3: 2, 2: 2}
KZ = {4: 2, 3: 3, 2: 4}


def fig_gate_counts(grid):
    g = grid[grid.topology == "all-to-all"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    for ax, col, title in zip(axes, ("total_gates", "two_qubit_gates"),
                              ("(a) Total transpiled gates", "(b) Two-qubit (CX) gates")):
        for k in (4, 3, 2):
            d = g[g.k_max == k].sort_values("qubits")
            ax.plot(d.qubits, d[col], marker=KMARK[k], ms=KMS[k], lw=KLW[k], color=KCOL[k],
                    label=KLAB[k], zorder=KZ[k])
        ax.set_xticks([2, 3, 4]); ax.set_xlabel("Number of qubits n")
        ax.set_ylabel("Gate count"); ax.set_title(title, loc="left")
    axes[0].legend(loc="upper left")
    fig.suptitle("Gate count versus qubit count (basis rz, sx, x, cx; optimisation level 1)",
                 x=0.01, ha="left", fontsize=11, y=1.03)
    save(fig, "fig07_gate_count_vs_qubits.png")


def fig_depth(grid):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=True)
    for ax, topo, title in zip(axes, ("all-to-all", "linear"),
                               ("(a) All-to-all connectivity", "(b) Linear nearest-neighbour chain")):
        g = grid[grid.topology == topo]
        for k in (4, 3, 2):
            d = g[g.k_max == k].sort_values("qubits")
            ax.plot(d.qubits, d.depth, marker=KMARK[k], ms=KMS[k], lw=KLW[k], color=KCOL[k],
                    label=KLAB[k], zorder=KZ[k])
        ax.set_xticks([2, 3, 4]); ax.set_xlabel("Number of qubits n"); ax.set_title(title, loc="left")
    axes[0].set_ylabel("Transpiled circuit depth")
    axes[0].legend(loc="upper left")
    fig.suptitle("Transpiled circuit depth versus qubit count", x=0.01, ha="left", fontsize=11, y=1.03)
    save(fig, "fig08_depth_vs_qubits.png")


def fig_fidelity_vs_noise(summ):
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.2), sharey=True)
    x = np.arange(3)
    for ax, n in zip(axes, (2, 3, 4)):
        d = summ[summ.qubits == n]
        for v in ("Exact", "AQFT m=3", "AQFT m=2"):
            dv = d[d.variant == v].set_index("noise").reindex(["N0", "N1", "N2"])
            if dv.dist_fidelity_mean.isna().all():
                continue
            ax.errorbar(x, dv.dist_fidelity_mean, yerr=dv.dist_fidelity_sd, marker="o", ms=6,
                        lw=2, capsize=3, color=VARIANT_COLOR[v], label=v)
        ax.set_xticks(x, ["N0\nideal", "N1\nlow", "N2\nmoderate"])
        ax.set_title(f"n = {n}", loc="left"); ax.set_xlabel("Noise level")
    axes[0].set_ylabel("Distribution fidelity (mean ± SD, 5 seeds)")
    axes[0].set_ylim(0.70, 1.01)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    fig.suptitle("Fidelity versus noise level (periodic-signal inputs, 1024 shots)",
                 x=0.01, ha="left", fontsize=11, y=1.03)
    save(fig, "fig09_fidelity_vs_noise.png")


def fig_exact_vs_approx(res):
    d = res[res.qubits == 4].set_index("variant").reindex(["Exact", "AQFT m=3", "AQFT m=2"])
    metrics = [("opt1_total_gates", "Total gates"), ("opt1_two_qubit_gates", "CX gates"),
               ("opt1_depth", "Depth")]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    w = 0.26; x = np.arange(len(metrics))
    for i, v in enumerate(d.index):
        vals = [d.loc[v, c] for c, _ in metrics]
        bars = ax.bar(x + (i - 1) * w, vals, w - 0.03, color=VARIANT_COLOR[v], label=v)
        for b, (c, _) in zip(bars, metrics):
            base = d.loc["Exact", c]
            txt = f"{int(b.get_height())}" if v == "Exact" else \
                (f"{int(b.get_height())}\n(−{100*(base-b.get_height())/base:.0f}%)" if b.get_height() < base
                 else f"{int(b.get_height())}\n(0%)")
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, txt,
                    ha="center", va="bottom", fontsize=8, color="#52514e")
    ax.set_xticks(x, [lab for _, lab in metrics]); ax.set_ylabel("Count")
    ax.set_ylim(0, 60); ax.legend(loc="upper right", ncol=3)
    ax.set_title("Exact versus approximate QFT resources, n = 4 (percent saving vs exact)",
                 loc="left")
    save(fig, "fig10_exact_vs_approx_resources.png")


def fig_noise_sweep(sweep):
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    for v in ("Exact", "AQFT m=3", "AQFT m=2"):
        d = sweep[sweep.variant == v]
        ax.errorbar(d.p2 * 100, d.dist_fidelity_mean, yerr=d.dist_fidelity_sd, marker="o", ms=4,
                    lw=2, capsize=2, color=VARIANT_COLOR[v], label=v)
    ex = sweep[sweep.variant == "Exact"].set_index("scale")
    m3 = sweep[sweep.variant == "AQFT m=3"].set_index("scale")
    diff = (m3.dist_fidelity_mean - ex.dist_fidelity_mean)
    s = diff.index.values; dv = diff.values
    i = np.where(np.diff(np.sign(dv)) > 0)[0]
    if len(i):
        j = i[0]; s_star = s[j] - dv[j] * (s[j + 1] - s[j]) / (dv[j + 1] - dv[j])
        ax.axvline(2 * s_star, color="#8a8984", lw=1, ls="--")
        ax.text(2 * s_star + 0.05, 0.98, f"exact / m=3 crossover\np₂ ≈ {2*s_star:.2f}%",
                fontsize=8.5, color="#52514e", va="top")
    ax.axvline(2.0, color="#c3c2b7", lw=1, ls=":")
    ax.text(2.03, 0.995, "N2 rates", fontsize=8.5, color="#52514e", va="top")
    ax.set_xlabel("Two-qubit depolarizing error p₂ (%)   [p₁ = p₂/2, readout = 1.5·p₂]")
    ax.set_ylabel("Distribution fidelity (mean ± SD)")
    ax.set_title("Noise-strength sweep, n = 4: accuracy-resource crossover", loc="left")
    ax.legend(loc="upper right")
    save(fig, "fig11_noise_sweep_crossover.png")
    return float(2 * s_star) if len(i) else None


def fig_distributions(counts, seed=11):
    n, N = 4, 16
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.6))
    k = np.arange(N); labels = [format(i, "04b") for i in k]
    # (a) comb r=4 o=1, exact QFT, three noise levels
    ax = axes[0]
    ref = reference_distribution(signal_vector(n, "comb", (4, 1)))
    w = 0.27
    for i, (eid, lvl) in enumerate([("E4", "N0"), ("E10", "N1"), ("E16", "N2")]):
        c = counts[f"{eid}|{seed}|comb|(4, 1)"]
        p = np.array([c.get(b, 0) for b in labels]) / 1024
        ax.bar(k + (i - 1) * w, p, w - 0.03, color=NOISE_COLOR[lvl], label=f"Exact QFT, {lvl}")
    ax.scatter(k, ref, marker="_", s=260, lw=2.2, color=REF_COLOR, zorder=5, label="Classical DFT |X_k|²")
    ax.set_xticks(k, labels, rotation=90, fontsize=8); ax.set_ylabel("Probability")
    ax.set_title("(a) Periodic comb (period r = 4, offset 1): exact QFT under N0 / N1 / N2", loc="left")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(0, 0.36)
    # (b) tone j=11 at N0: exact vs approximations
    ax = axes[1]
    ref = reference_distribution(signal_vector(n, "tone", (11,)))
    for i, (eid, v) in enumerate([("E4", "Exact"), ("E5", "AQFT m=3"), ("E6", "AQFT m=2")]):
        c = counts[f"{eid}|{seed}|tone|(11,)"]
        p = np.array([c.get(b, 0) for b in labels]) / 1024
        ax.bar(k + (i - 1) * w, p, w - 0.03, color=VARIANT_COLOR[v], label=f"{v}, N0")
    ax.scatter(k, ref, marker="_", s=260, lw=2.2, color=REF_COLOR, zorder=5, label="Classical DFT |X_k|²")
    ax.set_xticks(k, labels, rotation=90, fontsize=8)
    ax.set_xlabel("Measured outcome k (bitstring q₃q₂q₁q₀)"); ax.set_ylabel("Probability")
    ax.set_title("(b) Complex tone (frequency j = 11): exact versus approximate QFT, ideal noise", loc="left")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    ax.set_ylim(0, 1.12)
    fig.suptitle(f"Measurement probability distributions (1024 shots, seed {seed})",
                 x=0.01, ha="left", fontsize=11, y=1.0)
    fig.tight_layout()
    save(fig, "fig12_measurement_distributions.png")


# ---- statistics ----------------------------------------------------------
def welch_tests(per_seed: pd.DataFrame) -> pd.DataFrame:
    pairs = [("E10", "E11"), ("E10", "E12"), ("E16", "E17"), ("E16", "E18"),
             ("E4", "E5"), ("E14", "E15"), ("E8", "E9")]
    rows = []
    for a, b in pairs:
        xa = per_seed[per_seed.exp_id == a].dist_fidelity_mean.values
        xb = per_seed[per_seed.exp_id == b].dist_fidelity_mean.values
        t, p = stats.ttest_ind(xb, xa, equal_var=False)
        rows.append({"comparison": f"{b} vs {a}", "mean_a": xa.mean(), "mean_b": xb.mean(),
                     "difference_b_minus_a": xb.mean() - xa.mean(), "welch_t": t, "p_value": p})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "welch_tests.csv"), index=False)
    print(df.round(5).to_string(index=False))
    return df


def main():
    os.makedirs(FIG, exist_ok=True)
    summ = pd.read_csv(os.path.join(RES, "experiment_summary.csv"))
    res = pd.read_csv(os.path.join(RES, "resources.csv"))
    sweep = pd.read_csv(os.path.join(RES, "noise_sweep.csv"))
    per_seed = pd.read_csv(os.path.join(RES, "per_seed_results.csv"))
    with open(os.path.join(RES, "counts.json")) as fh:
        counts = json.load(fh)
    circuit_figures()
    grid = resource_grid()
    fig_gate_counts(grid)
    fig_depth(grid)
    fig_fidelity_vs_noise(summ)
    fig_exact_vs_approx(res)
    s_star = fig_noise_sweep(sweep)
    print(f"  crossover two-qubit error p2* = {s_star:.3f}%")
    fig_distributions(counts)
    welch_tests(per_seed)
    with open(os.path.join(RES, "crossover.json"), "w") as fh:
        json.dump({"p2_crossover_percent_exact_vs_m3": s_star}, fh, indent=2)


if __name__ == "__main__":
    main()
