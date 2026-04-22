"""
preprocess.py
-------------
Loads the raw semiconductor process dataset, validates it,
removes near-zero-variance features, and standardizes the data.

Outputs saved to:
  data/X_scaled.npy       — standardized feature matrix (numpy)
  data/feature_names.txt  — retained feature names

Usage:
    python scripts/preprocess.py
"""

import numpy as np
import pandas as pd
import os

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH     = os.path.join("data", "semiconductor_process_data.csv")
VAR_THRESHOLD = 1e-6   # drop columns with variance below this

# ── Load data ────────────────────────────────────────────────────────────────
print(f"[INFO] Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, index_col="run_id")
print(f"[INFO] Raw shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ── Basic validation ─────────────────────────────────────────────────────────
n_missing = df.isnull().sum().sum()
print(f"[INFO] Missing values: {n_missing}")
if n_missing > 0:
    print("[WARN] Dropping rows with any missing values.")
    df.dropna(inplace=True)

# ── Remove near-zero-variance features ───────────────────────────────────────
col_var    = df.var(axis=0)
low_var    = col_var[col_var < VAR_THRESHOLD].index
if len(low_var):
    print(f"[INFO] Dropping {len(low_var)} near-zero-variance features.")
    df.drop(columns=low_var, inplace=True)

print(f"[INFO] Shape after cleaning: {df.shape[0]} rows × {df.shape[1]} columns")

# ── Standardize (zero mean, unit variance) ───────────────────────────────────
# Using manual computation to avoid sklearn dependency leakage at this step.
X      = df.values.astype(np.float64)
mu     = X.mean(axis=0)
sigma  = X.std(axis=0, ddof=1)
sigma[sigma == 0] = 1.0   # guard against any remaining zero-std columns
X_scaled = (X - mu) / sigma

print(f"[INFO] Standardization complete. Mean≈{X_scaled.mean():.4f}, Std≈{X_scaled.std():.4f}")

# ── Save outputs ──────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)

np.save(os.path.join("data", "X_scaled.npy"),    X_scaled)
np.save(os.path.join("data", "X_mean.npy"),      mu)
np.save(os.path.join("data", "X_std.npy"),       sigma)

feature_names = df.columns.tolist()
with open(os.path.join("data", "feature_names.txt"), "w") as f:
    f.write("\n".join(feature_names))

print(f"[OK]   X_scaled saved  → data/X_scaled.npy  {X_scaled.shape}")
print(f"[OK]   Feature names   → data/feature_names.txt  ({len(feature_names)} features)")
