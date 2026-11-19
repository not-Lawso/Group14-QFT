# Group 14 — Quantum Fourier Transform: circuit complexity, approximation and noise

M. S. Ramaiah Institute of Technology · CSE (Cyber Security) · Quantum Computing (CYE732)

The project studies how the cost of the exact and approximate QFT changes for 2, 3 and 4 qubits, and how accurate each circuit is. Cost is measured as total gates, CX count and transpiled depth. Accuracy is measured against a classical DFT reference under ideal (N0), low (N1) and moderate (N2) noise in Qiskit Aer.

## Run

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run_experiments.py     # ~25 s: validation, resources, E1–E18, noise sweep S1 -> results/
python make_figures.py        # figures/ + results/welch_tests.csv, resource_grid.csv, crossover.json
python env_check.py           # print the software environment
python show_results.py        # print the E1–E18 summary table
```

The results are deterministic. Simulator seeds are 11, 23, 37, 51 and 73, and the transpiler seed is 2026. Every run uses 1024 shots.

## Layout

| File | Purpose |
|---|---|
| `qft_study/circuits.py` | `build_qft(n, m, do_swaps)` builds the exact or approximate QFT (it keeps CP(2π/2^k) for k ≤ m). Also prepares the basis-state, complex-tone and periodic-comb inputs. |
| `qft_study/dft.py` | Classical DFT baseline (NumPy FFT, same +2πi convention), reference distributions and operation counts. |
| `qft_study/noise.py` | N0/N1/N2 noise models: depolarizing error on `sx`/`x` and `cx`, symmetric readout error, attached to `AerSimulator`. |
| `qft_study/metrics.py` | One shared transpilation method (basis rz/sx/x/cx, optimisation level 1), resource counts, and the distribution, state and process fidelities. |
| `qft_study/experiments.py` | Validation, the experiment matrix, repeated runs, aggregation (mean ± SD) and the S1 sweep. |
| `run_experiments.py`, `make_figures.py` | Pipeline entry points. |
| `results/` | Results written by the pipeline (CSV/JSON); `counts.json` holds every raw count dictionary. |

## Notes

- The approximation parameter is `m` (`k_max` in the report). It equals `n − approximation_degree` in Qiskit's `synth_qft_full`, and `run_experiments.py` checks this equivalence on every run.
- `rz` is a virtual gate and is modelled as error-free. Thermal relaxation is not modelled.
- All results come from simulation. No hardware runs were made.
