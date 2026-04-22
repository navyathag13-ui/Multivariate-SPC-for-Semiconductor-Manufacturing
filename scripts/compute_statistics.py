"""
compute_statistics.py
---------------------
Computes multivariate SPC monitoring statistics:
  - Hotelling's T²  : measures deviation within the PC subspace
  - SPE / Q residual: measures deviation outside the PC subspace

Both statistics are computed for every observation.
Upper control limits (UCL) are set at 95% confidence using chi-squared approximation.
Flagged observations (exceeding either UCL) are saved as an anomaly summary.

Outputs saved to:
  outputs/t2_values.npy          — Hotelling's T² for each observation
  outputs/spe_values.npy         — SPE values for each observation
  outputs/t2_ucl.txt             — T² upper control limit
  outputs/spe_ucl.txt            — SPE upper control limit
  outputs/anomaly_flags.npy      — boolean array: True = flagged anomaly
  outputs/anomaly_summary.csv    — table of flagged run IDs and statistics

Usage:
    python scripts/compute_statistics.py
"""

import numpy as np
import pandas as pd
import os
import sys

# Allow import from scripts/ directory when run from project root
sys.path.insert(0, os.path.dirname(__file__))
from utils import chi2_threshold, spe_threshold, flag_anomalies, print_summary

# ── Significance level ────────────────────────────────────────────────────────
ALPHA = 0.05

# ── Load PCA results ──────────────────────────────────────────────────────────
scores_ret    = np.load(os.path.join("data", "pca_scores_ret.npy"))   # (N, k)
scores_all    = np.load(os.path.join("data", "pca_scores_all.npy"))   # (N, p)
components    = np.load(os.path.join("data", "pca_components.npy"))   # (p, p)
eigenvalues   = np.load(os.path.join("data", "pca_eigenvalues.npy"))  # (p,)
X_scaled      = np.load(os.path.join("data", "X_scaled.npy"))         # (N, p)

with open(os.path.join("data", "n_components.txt")) as f:
    k = int(f.read().strip())

print(f"[INFO] Loaded: {scores_ret.shape[0]} observations, {k} retained PCs")

# ── 1. Hotelling's T² ─────────────────────────────────────────────────────────
# T²_i = sum_j ( t_ij² / lambda_j )
# where t_ij = score of obs i on PC j
#       lambda_j = eigenvalue (variance explained by PC j)
lambdas = eigenvalues[:k]
T2 = np.sum((scores_ret ** 2) / lambdas, axis=1)

# UCL: chi-squared approximation with k degrees of freedom
T2_UCL = chi2_threshold(ALPHA, k)
print(f"[INFO] T² UCL ({(1-ALPHA)*100:.0f}%, df={k}): {T2_UCL:.4f}")

# ── 2. SPE / Q residual ───────────────────────────────────────────────────────
# SPE_i = || x_i - x̂_i ||²
# where x̂_i = reconstructed observation using k retained PCs
P_ret     = components[:k, :].T          # (p, k) — retained loading matrix
X_hat     = scores_ret @ P_ret.T         # (N, p) — reconstruction
residuals = X_scaled - X_hat             # (N, p) — residual space
SPE       = np.sum(residuals ** 2, axis=1)

# UCL: Jackson-Mudholkar chi-squared approximation
SPE_UCL = spe_threshold(SPE, ALPHA)
print(f"[INFO] SPE UCL ({(1-ALPHA)*100:.0f}%): {SPE_UCL:.4f}")

# ── 3. Flag anomalies ─────────────────────────────────────────────────────────
flags = flag_anomalies(T2, SPE, T2_UCL, SPE_UCL)
print_summary(T2, SPE, T2_UCL, SPE_UCL)

# ── 4. Build anomaly summary table ────────────────────────────────────────────
flagged_indices = np.where(flags)[0]
anomaly_df = pd.DataFrame({
    "run_id"    : flagged_indices,
    "T2"        : T2[flagged_indices],
    "SPE"       : SPE[flagged_indices],
    "T2_exceed" : T2[flagged_indices] > T2_UCL,
    "SPE_exceed": SPE[flagged_indices] > SPE_UCL,
}).set_index("run_id")
anomaly_df.sort_values("T2", ascending=False, inplace=True)

# ── 5. Save outputs ───────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

np.save(os.path.join("outputs", "t2_values.npy"),    T2)
np.save(os.path.join("outputs", "spe_values.npy"),   SPE)
np.save(os.path.join("outputs", "anomaly_flags.npy"), flags.astype(np.uint8))

with open(os.path.join("outputs", "t2_ucl.txt"),  "w") as f: f.write(str(T2_UCL))
with open(os.path.join("outputs", "spe_ucl.txt"), "w") as f: f.write(str(SPE_UCL))

anomaly_df.to_csv(os.path.join("outputs", "anomaly_summary.csv"))

print(f"[OK]   T² values         → outputs/t2_values.npy")
print(f"[OK]   SPE values        → outputs/spe_values.npy")
print(f"[OK]   Anomaly summary   → outputs/anomaly_summary.csv  ({len(anomaly_df)} flagged runs)")
