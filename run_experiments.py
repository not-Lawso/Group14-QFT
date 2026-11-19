"""Group 14 - QFT resource and accuracy study.  Runs the full pipeline:

    python run_experiments.py           # experiments + CSV/JSON results
    python make_figures.py              # figures from the saved results

Outputs are written to ./results.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import qiskit
import qiskit_aer

from qft_study.experiments import (SEEDS, SHOTS, NOISE_LEVELS, resource_analysis, run_noise_sweep,
                                   run_all_experiments, save_counts,
                                   validate_implementation)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def environment_info() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "qiskit_aer": qiskit_aer.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "shots": SHOTS,
        "seeds": SEEDS,
        "noise_levels": {k: v for k, v in NOISE_LEVELS.items()},
        "cpu_count": os.cpu_count(),
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()
    env = environment_info()
    print("=" * 72)
    print("Group 14 | Quantum Fourier Transform | environment")
    print("=" * 72)
    for k in ("python", "qiskit", "qiskit_aer", "numpy", "pandas", "matplotlib"):
        print(f"  {k:<11s}: {env[k]}")
    print(f"  shots      : {SHOTS}    seeds: {SEEDS}")

    print("\n[1/3] Validating circuits against the classical DFT")
    checks = validate_implementation()
    ok = all(v for k, v in checks.items() if k.endswith(tuple(f"degree{i}" for i in range(1, 4))))
    print(f"  AQFT construction equivalent to qiskit.synthesis.synth_qft_full: {ok}")

    print("\n[2/3] Resource analysis (transpiled to rz, sx, x, cx)")
    res_df = resource_analysis()
    res_df.to_csv(os.path.join(OUT, "resources.csv"), index=False)

    print("\n[3/3] Running experiments E1-E18 (1024 shots x 5 seeds each)")
    summary, per_seed, per_input, counts = run_all_experiments()
    summary.to_csv(os.path.join(OUT, "experiment_summary.csv"), index=False)
    per_seed.to_csv(os.path.join(OUT, "per_seed_results.csv"), index=False)
    per_input.to_csv(os.path.join(OUT, "per_input_results.csv"), index=False)
    save_counts(counts, os.path.join(OUT, "counts.json"))

    print("\n[S1] Supplementary noise-strength sweep (n = 4, s x N2 rates)")
    sweep = run_noise_sweep()
    sweep.to_csv(os.path.join(OUT, "noise_sweep.csv"), index=False)

    env["runtime_seconds"] = round(time.time() - t_start, 1)
    env["validation"] = checks
    with open(os.path.join(OUT, "environment.json"), "w") as fh:
        json.dump(env, fh, indent=2, default=str)
    print(f"\nDone in {env['runtime_seconds']} s. Results written to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
