"""
plot_results.py
---------------
Generates all engineering plots for the multivariate SPC analysis.

Plots generated (saved to plots/):
  01_scree_plot.png              — eigenvalue scree plot
  02_cumulative_variance.png     — cumulative explained variance
  03_score_plot_pc1_pc2.png      — PC1 vs PC2 score plot (normal vs flagged)
  04_t2_control_chart.png        — Hotelling's T² control chart
  05_spe_control_chart.png       — SPE / Q residual control chart
  06_anomaly_summary_bar.png     — count of flags by detection type
  07_pairwise_pca.png            — pairwise PC score plot (first 3 PCs)

Usage:
    python scripts/plot_results.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Load all required data ────────────────────────────────────────────────────
explained_var = np.load(os.path.join("data", "pca_explained_var.npy"))
eigenvalues   = np.load(os.path.join("data", "pca_eigenvalues.npy"))
scores_ret    = np.load(os.path.join("data", "pca_scores_ret.npy"))
T2            = np.load(os.path.join("outputs", "t2_values.npy"))
SPE           = np.load(os.path.join("outputs", "spe_values.npy"))
flags         = np.load(os.path.join("outputs", "anomaly_flags.npy")).astype(bool)

with open(os.path.join("data",    "n_components.txt"))  as f: k = int(f.read())
with open(os.path.join("outputs", "t2_ucl.txt"))        as f: T2_UCL  = float(f.read())
with open(os.path.join("outputs", "spe_ucl.txt"))       as f: SPE_UCL = float(f.read())

os.makedirs("plots", exist_ok=True)

cumvar = np.cumsum(explained_var)
N      = len(T2)
runs   = np.arange(N)

# ── Shared style ──────────────────────────────────────────────────────────────
NORMAL_COLOR  = "#2196F3"   # blue
ANOMALY_COLOR = "#F44336"   # red
UCL_COLOR     = "#FF9800"   # orange
GRID_ALPHA    = 0.3
FIG_DPI       = 150

def save(fig, name):
    path = os.path.join("plots", name)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]   Saved → {path}")

# ────────────────────────────────────────────────────────────────────────────
# Plot 01 — Scree plot
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
n_show = min(30, len(eigenvalues))
ax.plot(range(1, n_show + 1), eigenvalues[:n_show], "o-",
        color=NORMAL_COLOR, linewidth=2, markersize=6)
ax.axvline(k, color=UCL_COLOR, linestyle="--", linewidth=1.5,
           label=f"Retained PCs = {k}")
ax.set_xlabel("Principal Component", fontsize=12)
ax.set_ylabel("Eigenvalue", fontsize=12)
ax.set_title("PCA Scree Plot", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=GRID_ALPHA)
ax.set_xticks(range(1, n_show + 1))
save(fig, "01_scree_plot.png")

# ────────────────────────────────────────────────────────────────────────────
# Plot 02 — Cumulative explained variance
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
n_show2 = min(50, len(cumvar))
ax.plot(range(1, n_show2 + 1), cumvar[:n_show2] * 100, "s-",
        color=NORMAL_COLOR, linewidth=2, markersize=5)
ax.axhline(80, color=UCL_COLOR, linestyle="--", linewidth=1.5,
           label="80% variance threshold")
ax.axvline(k, color=ANOMALY_COLOR, linestyle=":", linewidth=1.5,
           label=f"PC {k}: {cumvar[k-1]*100:.1f}% cumulative")
ax.set_xlabel("Number of Principal Components", fontsize=12)
ax.set_ylabel("Cumulative Explained Variance (%)", fontsize=12)
ax.set_title("Cumulative Explained Variance — PCA", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=GRID_ALPHA)
ax.set_ylim([0, 105])
save(fig, "02_cumulative_variance.png")

# ────────────────────────────────────────────────────────────────────────────
# Plot 03 — Score plot PC1 vs PC2
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
normal_mask  = ~flags
ax.scatter(scores_ret[normal_mask,  0], scores_ret[normal_mask,  1],
           c=NORMAL_COLOR,  s=12, alpha=0.5, label="Normal")
ax.scatter(scores_ret[flags,         0], scores_ret[flags,         1],
           c=ANOMALY_COLOR, s=30, alpha=0.8, marker="^", label="Flagged Anomaly")
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel(f"PC 1 ({explained_var[0]*100:.1f}%)", fontsize=12)
ax.set_ylabel(f"PC 2 ({explained_var[1]*100:.1f}%)", fontsize=12)
ax.set_title("PCA Score Plot — PC1 vs PC2", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=GRID_ALPHA)
save(fig, "03_score_plot_pc1_pc2.png")

# ────────────────────────────────────────────────────────────────────────────
# Plot 04 — Hotelling's T² control chart
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
t2_exceed = T2 > T2_UCL
ax.plot(runs[~t2_exceed], T2[~t2_exceed], ".", color=NORMAL_COLOR,
        markersize=4, alpha=0.6, label="Normal")
ax.plot(runs[t2_exceed],  T2[t2_exceed],  "^", color=ANOMALY_COLOR,
        markersize=6, alpha=0.9, label="T² Exceedance")
ax.axhline(T2_UCL, color=UCL_COLOR, linewidth=2, linestyle="--",
           label=f"UCL = {T2_UCL:.2f} (95%)")
ax.set_xlabel("Run Index", fontsize=12)
ax.set_ylabel("Hotelling's T²", fontsize=12)
ax.set_title("Hotelling's T² Control Chart", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=GRID_ALPHA)
save(fig, "04_t2_control_chart.png")

# ────────────────────────────────────────────────────────────────────────────
# Plot 05 — SPE / Q residual control chart
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
spe_exceed = SPE > SPE_UCL
ax.plot(runs[~spe_exceed], SPE[~spe_exceed], ".", color=NORMAL_COLOR,
        markersize=4, alpha=0.6, label="Normal")
ax.plot(runs[spe_exceed],  SPE[spe_exceed],  "^", color=ANOMALY_COLOR,
        markersize=6, alpha=0.9, label="SPE Exceedance")
ax.axhline(SPE_UCL, color=UCL_COLOR, linewidth=2, linestyle="--",
           label=f"UCL = {SPE_UCL:.4f} (95%)")
ax.set_xlabel("Run Index", fontsize=12)
ax.set_ylabel("SPE (Q Residual)", fontsize=12)
ax.set_title("SPE / Q Residual Control Chart", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=GRID_ALPHA)
save(fig, "05_spe_control_chart.png")

# ────────────────────────────────────────────────────────────────────────────
# Plot 06 — Anomaly summary bar chart
# ────────────────────────────────────────────────────────────────────────────
t2_only  = ((T2 > T2_UCL) & (SPE <= SPE_UCL)).sum()
spe_only = ((T2 <= T2_UCL) & (SPE > SPE_UCL)).sum()
both     = ((T2 > T2_UCL) & (SPE > SPE_UCL)).sum()
normal   = (~flags).sum()

categories = ["Normal", "T² Only", "SPE Only", "Both T² & SPE"]
counts     = [normal, t2_only, spe_only, both]
colors     = [NORMAL_COLOR, "#9C27B0", "#FF5722", ANOMALY_COLOR]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(categories, counts, color=colors, edgecolor="white", linewidth=0.8)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            str(cnt), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Number of Observations", fontsize=12)
ax.set_title("Anomaly Detection Summary by Category", fontsize=14, fontweight="bold")
ax.grid(True, axis="y", alpha=GRID_ALPHA)
ax.set_ylim([0, max(counts) * 1.15])
save(fig, "06_anomaly_summary_bar.png")

# ────────────────────────────────────────────────────────────────────────────
# Plot 07 — Pairwise PC score plot (PC1, PC2, PC3)
# ────────────────────────────────────────────────────────────────────────────
n_pair = min(3, k)   # up to 3 PCs
fig, axes = plt.subplots(n_pair, n_pair, figsize=(10, 9))
fig.suptitle("Pairwise PCA Score Plot — Normal vs Flagged Anomalies",
             fontsize=13, fontweight="bold", y=1.01)

for r in range(n_pair):
    for c in range(n_pair):
        ax = axes[r][c] if n_pair > 1 else axes
        if r == c:
            ax.hist(scores_ret[normal_mask, r], bins=40,
                    color=NORMAL_COLOR, alpha=0.6, density=True)
            ax.hist(scores_ret[flags,       r], bins=20,
                    color=ANOMALY_COLOR, alpha=0.7, density=True)
            ax.set_title(f"PC {r+1}", fontsize=10)
        else:
            ax.scatter(scores_ret[normal_mask, c], scores_ret[normal_mask, r],
                       s=5, c=NORMAL_COLOR, alpha=0.4)
            ax.scatter(scores_ret[flags, c],        scores_ret[flags, r],
                       s=20, c=ANOMALY_COLOR, alpha=0.8, marker="^")
            ax.set_xlabel(f"PC {c+1}", fontsize=9)
            ax.set_ylabel(f"PC {r+1}", fontsize=9)
        ax.grid(True, alpha=GRID_ALPHA)

p1 = mpatches.Patch(color=NORMAL_COLOR,  label="Normal")
p2 = mpatches.Patch(color=ANOMALY_COLOR, label="Flagged")
fig.legend(handles=[p1, p2], loc="upper right", fontsize=10)
fig.tight_layout()
save(fig, "07_pairwise_pca.png")

print("\n[DONE] All plots saved to plots/")
