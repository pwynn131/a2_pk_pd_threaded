import pandas as pd
import numpy as np

dir = "output_2h_A_999999_wu_2500_alpha_98"
df = pd.read_csv(f"{dir}/500_pats_122_points_PATIENT_2h_A_999999_wu_2500_alpha_98.csv")

summary_rows = []
for p in ["A", "c", "gamma"]:
    sub_df = df[df['parameter'] == p]
    avg_rhat = np.mean(sub_df["r_hat"])
    mae = np.mean(sub_df["AE"])
    coverage = np.mean(sub_df["w_in_interval"])
    summary_rows.append({
        "Parameter": p,
        "Avg_r_hat": avg_rhat,
        "MAE": mae,
        "Coverage (%)": coverage
    })
results = pd.DataFrame(summary_rows)
results.to_csv(f"{dir}/results.csv")

