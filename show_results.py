"""Print the aggregated experiment table (E1-E18) from results/experiment_summary.csv."""
import pandas as pd

df = pd.read_csv("results/experiment_summary.csv")
df["F_dist (mean±SD)"] = df.apply(lambda r: f"{r.dist_fidelity_mean:.4f} ± {r.dist_fidelity_sd:.4f}", axis=1)
df["F_state"] = df.state_fidelity_all.map(lambda v: f"{v:.4f}")
cols = ["exp_id", "qubits", "variant", "noise", "shots", "seeds", "total_gates",
        "two_qubit_gates", "depth", "F_dist (mean±SD)", "F_state"]
print(df[cols].rename(columns={"two_qubit_gates": "CX", "total_gates": "gates"}).to_string(index=False))
