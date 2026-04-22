"""
run_pca.py
----------
Performs PCA on the standardized process data.

Steps:
  1. Load X_scaled from preprocessing step.
  2. Fit PCA (full decomposition).
  3. Determine the number of PCs needed to explain ≥80% of total variance.
  4. Save PCA model components and scores for downstream analysis.
  5. Print an explained variance summary table.

Outputs saved to:
  data/pca_components.npy       — principal component loadings (eigenvectors)
  data/pca_explained_var.npy    — explained variance ratios
  data/pca_scores.npy           — scores for all PCs
  data/n_components.txt         — number of retained components
  data/pca_eigenvalues.npy      — eigenvalues (for scree plot)

Usage:
    python scripts/run_pca.py
"""

import numpy as np
from sklearn.decomposition import PCA
import os

# ── Load preprocessed data ───────────────────────────────────────────────────
X_scaled = np.load(os.path.join("data", "X_scaled.npy"))
print(f"[INFO] Loaded X_scaled: {X_scaled.shape}")

# ── Fit full PCA ─────────────────────────────────────────────────────────────
print("[INFO] Fitting PCA (full decomposition)...")
pca = PCA(random_state=42)
pca.fit(X_scaled)

explained_var_ratio   = pca.explained_variance_ratio_
cumulative_var        = np.cumsum(explained_var_ratio)
eigenvalues           = pca.explained_variance_

# ── Select number of components ───────────────────────────────────────────────
VARIANCE_TARGET = 0.80
n_components = int(np.searchsorted(cumulative_var, VARIANCE_TARGET)) + 1
print(f"[INFO] Components needed for ≥{VARIANCE_TARGET*100:.0f}% variance: {n_components}")
print(f"[INFO] Actual cumulative variance at {n_components} PCs: "
      f"{cumulative_var[n_components-1]*100:.2f}%")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n── Explained Variance per PC (top components) ────────────────")
print(f"{'PC':>4}  {'Eigenvalue':>12}  {'Ind. Var %':>10}  {'Cum. Var %':>10}")
print("-" * 46)
for i in range(min(n_components + 5, len(eigenvalues))):
    marker = " ◄" if i + 1 == n_components else ""
    print(f"{i+1:>4}  {eigenvalues[i]:>12.4f}  "
          f"{explained_var_ratio[i]*100:>9.2f}%  "
          f"{cumulative_var[i]*100:>9.2f}%{marker}")
print("──────────────────────────────────────────────────────────────\n")

# ── Compute scores for ALL PCs (needed for T² and SPE) ───────────────────────
scores_all = pca.transform(X_scaled)           # (N_OBS, N_VARS)
scores_ret = scores_all[:, :n_components]      # (N_OBS, n_components)

# ── Save outputs ──────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)

np.save(os.path.join("data", "pca_components.npy"),    pca.components_)
np.save(os.path.join("data", "pca_explained_var.npy"), explained_var_ratio)
np.save(os.path.join("data", "pca_eigenvalues.npy"),   eigenvalues)
np.save(os.path.join("data", "pca_scores_all.npy"),    scores_all)
np.save(os.path.join("data", "pca_scores_ret.npy"),    scores_ret)
np.save(os.path.join("data", "pca_mean.npy"),          pca.mean_)

with open(os.path.join("data", "n_components.txt"), "w") as f:
    f.write(str(n_components))

print(f"[OK]   PCA components saved → data/pca_components.npy")
print(f"[OK]   Retained scores     → data/pca_scores_ret.npy  {scores_ret.shape}")
print(f"[OK]   n_components        → data/n_components.txt  ({n_components})")
