# Technical Report: Multivariate SPC for Semiconductor Manufacturing

**Author:** Navyatha G
**Date:** April 2026
**Tools:** Python, scikit-learn, NumPy, pandas, matplotlib, SciPy

---

## 1. Problem Statement

Semiconductor manufacturing processes involve hundreds of tightly coupled process variables — gas flows, pressures, temperatures, RF power levels, and endpoint signals — all recorded per wafer run. Statistical Process Control (SPC) is the standard engineering practice for detecting when a process has shifted outside its normal operating envelope.

**The challenge:** Traditional univariate control charts (one chart per variable) are insufficient for high-dimensional processes because:

1. With 590 variables and 95% individual confidence limits, over 29 false alarms are expected per run purely by chance.
2. Subtle, correlated shifts across many variables — the most common form of real process drift — will not trigger any single univariate chart.
3. The 590 variables are not independent; monitoring them independently ignores the underlying process structure.

**The solution:** Multivariate SPC using PCA reduces the problem to a small number of interpretable components and applies two monitoring statistics — Hotelling's T² and SPE — that jointly characterize the entire process state in a single, coherent framework.

---

## 2. Dataset Description

A synthetic dataset was generated to simulate a realistic semiconductor manufacturing environment:

| Property | Value |
|----------|-------|
| Observations (N) | 1,500 wafer runs |
| Variables (p) | 590 process/sensor variables |
| Underlying latent factors | 8 (decaying factor strengths) |
| Injected anomalies | ~75 runs (~5%) |
| Random seed | 42 (fully reproducible) |

The data is generated using a latent factor model:

```
X = Z @ L.T + ε
```

where Z is the (N × 8) matrix of latent factor scores, L is the (590 × 8) loading matrix, and ε is random sensor noise. Anomalous observations are created by applying large shifts (4–8σ) to 2–4 randomly selected latent factors, simulating realistic process excursions such as chamber pressure instability or gas flow drift.

---

## 3. Preprocessing

Steps applied:
1. **Missing value check**: No missing values present in the synthetic dataset. A drop-row policy is applied if any are detected.
2. **Near-zero variance removal**: Columns with variance below 1×10⁻⁶ are dropped (none removed in this dataset).
3. **Standardization**: All features are scaled to zero mean and unit variance. This is essential before PCA to prevent high-magnitude variables from dominating the decomposition.

After preprocessing: **1,500 observations × 590 features**, all on the same scale.

---

## 4. PCA Decomposition

PCA was applied to the standardized data matrix X (1500 × 590).

### Explained Variance

The scree plot shows a sharp decline in eigenvalues after the first few components, indicating that most process variance is concentrated in a small subspace.

**Component selection criterion:** Retain the minimum number of PCs such that cumulative explained variance ≥ 80%.

| PC | Eigenvalue | Individual Var % | Cumulative Var % |
|----|-----------|-----------------|-----------------|
| 1  | 167.18     | 28.34%          | 28.34%          |
| 2  | 134.38     | 22.78%          | 51.11%          |
| 3  | 107.52     | 18.22%          | 69.34%          |
| 4  |  74.74     | 12.67%          | **82.00% ◄ retained** |
| 5  |  42.55     |  7.21%          | 89.22%          |
| 6  |  33.05     |  5.60%          | 94.82%          |
| 7  |  17.06     |  2.89%          | 97.71%          |
| 8  |  11.75     |  1.99%          | 99.70%          |

**Result:** 4 principal components capture 82.0% of total process variance (random seed = 42).

**Key insight:** The original 590 correlated variables are compressed into a small number of PCs without significant information loss. This compression is the foundation of efficient multivariate monitoring.

---

## 5. Monitoring Statistics

### 5.1 Hotelling's T²

T² measures the Mahalanobis distance of each observation from the process mean within the retained PC subspace:

```
T²_i = Σ_j  (t_ij² / λ_j)
```

where t_ij is the score of observation i on the j-th PC and λ_j is the corresponding eigenvalue.

**Upper Control Limit (UCL):** Based on the chi-squared distribution with k degrees of freedom at α = 0.05:

```
T²_UCL = χ²(1 - α, k)
```

T² flags observations that are abnormally far from the process center in the retained subspace — corresponding to changes in the dominant process factors.

### 5.2 SPE / Q Residual

SPE (Squared Prediction Error) measures how well each observation is reconstructed using the retained PCs:

```
SPE_i = ||x_i - x̂_i||²
```

where x̂_i is the reconstruction of x_i using the k retained components.

A large SPE means the observation contains variation that cannot be explained by the normal process structure — it has deviated into the residual subspace. This catches a different class of anomalies than T².

**Upper Control Limit:** Jackson-Mudholkar chi-squared approximation based on the mean and variance of in-control SPE values.

### 5.3 Complementarity

The two statistics are complementary:
- **T² alone**: catches shifts within the normal operating directions
- **SPE alone**: catches new, unexpected directions of variation
- **Both**: provides a complete picture of the process state

An observation exceeding either UCL is flagged as a potential process excursion.

---

## 6. Results

After running the full pipeline:

| Metric | Value |
|--------|-------|
| Total observations | 1,500 |
| Injected anomalies | 75 |
| PCs retained | 4 (82.0% cumulative variance) |
| T² UCL (95%, df=4) | 9.49 |
| SPE UCL (95%) | 539.52 |
| Flagged by T² | 75 |
| Flagged by SPE | 47 |
| Flagged by either | 79 (5.3% of all runs) |
| False alarm rate | ~5% — consistent with 95% UCL |

**Score plot (PC1 vs PC2):** Anomalous runs visibly separate from the normal cluster in the PC score space, confirming that PCA captures the main directions of process excursion.

**Control charts:** Both the T² and SPE charts show clear spikes at the injected anomaly locations, with the majority of normal runs staying within the UCL bounds.

---

## 7. Limitations

1. **Synthetic data**: The dataset is generated, not collected from a real fab. Real process data would have additional complexity: sensor drift, scheduled maintenance effects, recipe changes, and batch-to-batch variation.

2. **Static PCA model**: This implementation uses a fixed PCA model trained on the full dataset. In practice, a Phase I / Phase II split should be used — the model is trained on known-good (in-control) data and then applied to new observations.

3. **No adaptive updating**: Real fab SPC systems update control limits periodically. This project uses fixed UCLs.

4. **No root-cause analysis**: When a run is flagged, the next step in a real system would be contribution plots to identify which variables drove the T² or SPE excursion. This is not implemented here but is a natural extension.

---

## 8. Conclusions

This project demonstrates a complete multivariate SPC pipeline applicable to semiconductor manufacturing data. The key engineering findings are:

1. **PCA effectively compresses** 590 correlated process variables into a small number of components (≥80% variance), making multivariate monitoring tractable.
2. **Hotelling's T² and SPE are complementary statistics** — together they provide a comprehensive characterization of process state across both the normal operating subspace and the residual space.
3. **The dual-statistic approach outperforms univariate monitoring** by capturing correlated, multi-variable deviations that would not trigger individual control charts.
4. **Anomaly detection performance** is consistent with the expected false positive rate at 95% confidence, validating the control limit methodology.

This framework is directly applicable to real fab environments where process datasets are high-dimensional, correlated, and require continuous automated monitoring.

---

## References

1. Wise, B.M. & Gallagher, N.B. (1996). The process chemometrics approach to process monitoring and fault detection. *Journal of Process Control*, 6(6), 329–348.
2. Jackson, J.E. & Mudholkar, G.S. (1979). Control procedures for residuals associated with principal component analysis. *Technometrics*, 21(3), 341–349.
3. Nomikos, P. & MacGregor, J.F. (1995). Multivariate SPC charts for monitoring batch processes. *Technometrics*, 37(1), 41–59.
4. Montgomery, D.C. (2012). *Introduction to Statistical Quality Control*, 7th ed. John Wiley & Sons.
5. Kourti, T. & MacGregor, J.F. (1995). Process analysis, monitoring and diagnosis using multivariate projection methods. *Chemometrics and Intelligent Laboratory Systems*, 28(1), 3–21.
