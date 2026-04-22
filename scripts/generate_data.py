"""
generate_data.py
----------------
Generates a synthetic semiconductor manufacturing process dataset.

Dataset characteristics:
  - 1500 observations (wafer runs)
  - 590 sensor/process variables (sensor_001 ... sensor_590)
  - Majority normal process runs with realistic correlations
  - ~5% injected anomalous runs representing process excursions

The data is saved to: data/semiconductor_process_data.csv
A label file is saved to: data/true_labels.csv  (for validation only)

Usage:
    python scripts/generate_data.py
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ── Dataset parameters ───────────────────────────────────────────────────────
N_OBS       = 1500   # total wafer runs
N_VARS      = 590    # sensor / process variables
N_FACTORS   = 8      # underlying latent process factors
ANOMALY_FRAC = 0.05  # fraction of runs with injected anomalies
N_ANOMALIES  = int(N_OBS * ANOMALY_FRAC)  # 75 anomalous runs

print(f"[INFO] Generating dataset: {N_OBS} observations × {N_VARS} variables")
print(f"[INFO] Injecting {N_ANOMALIES} anomalous observations (~{ANOMALY_FRAC*100:.0f}%)")

# ── Step 1: Generate correlated normal process data ──────────────────────────
# Latent factor model: X = Z @ L.T + noise
# Z : (N_OBS, N_FACTORS)  — latent scores
# L : (N_VARS, N_FACTORS) — loading matrix

Z_normal = rng.standard_normal((N_OBS, N_FACTORS))

# Loading matrix with decaying factor strengths
factor_strengths = np.array([5.0, 4.2, 3.5, 2.8, 2.2, 1.8, 1.3, 1.0])
L = rng.standard_normal((N_VARS, N_FACTORS)) * factor_strengths

# Sensor measurement noise
noise = rng.standard_normal((N_OBS, N_VARS)) * 0.5

X = Z_normal @ L.T + noise

# ── Step 2: Inject anomalies ─────────────────────────────────────────────────
# Anomalies are created by shifting latent factors beyond normal operating range.
# This simulates real process excursions (e.g., chamber pressure drift, temp spike).

anomaly_indices = rng.choice(N_OBS, size=N_ANOMALIES, replace=False)

for idx in anomaly_indices:
    # Randomly perturb 2–4 latent factors significantly
    n_affected = rng.integers(2, 5)
    affected_factors = rng.choice(N_FACTORS, size=n_affected, replace=False)
    shift = rng.choice([-1, 1], size=n_affected) * rng.uniform(4.0, 8.0, size=n_affected)
    Z_anomaly = Z_normal[idx].copy()
    Z_anomaly[affected_factors] += shift
    X[idx] = Z_anomaly @ L.T + rng.standard_normal(N_VARS) * 0.5

# ── Step 3: Build column names and DataFrame ─────────────────────────────────
col_names = [f"sensor_{i:03d}" for i in range(1, N_VARS + 1)]
df = pd.DataFrame(X, columns=col_names)
df.index.name = "run_id"

# ── Step 4: Build true labels (for post-hoc validation only) ─────────────────
labels = pd.Series(0, index=range(N_OBS), name="is_anomaly")
labels.iloc[anomaly_indices] = 1
labels.index.name = "run_id"

# ── Step 5: Save to disk ──────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)

data_path   = os.path.join("data", "semiconductor_process_data.csv")
labels_path = os.path.join("data", "true_labels.csv")

df.to_csv(data_path)
labels.to_csv(labels_path)

print(f"[OK]   Data saved  → {data_path}  ({df.shape[0]} rows × {df.shape[1]} cols)")
print(f"[OK]   Labels saved → {labels_path}")
print(f"       Normal runs  : {(labels == 0).sum()}")
print(f"       Anomalous runs: {(labels == 1).sum()}")
