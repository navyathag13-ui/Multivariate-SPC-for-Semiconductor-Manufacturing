# Interview Talking Points — Multivariate SPC for Semiconductor Manufacturing

---

## 30-Second Explanation

> "I built a multivariate statistical process control pipeline in Python for semiconductor manufacturing data. The dataset has 1,500 wafer runs and 590 process variables, which are too many to monitor individually. I used PCA to reduce the dimensionality down to about 5 principal components that capture 80% of the process variance, then computed two monitoring statistics — Hotelling's T² and SPE — to detect abnormal runs. The idea is that T² flags deviations within the normal process space, while SPE catches deviations that fall completely outside what the model expects. Together they flagged the injected process excursions with a false positive rate consistent with the 95% confidence limits I set."

---

## 60-Second Explanation

> "In semiconductor fabs, every wafer run generates hundreds of sensor readings — pressures, temperatures, gas flows, RF power levels, and so on. The traditional approach is to put a control chart on each variable separately, but that breaks down when you have 590 correlated variables. You'd get over 29 false alarms per run just by chance, and you'd still miss real process drifts that manifest as subtle, correlated shifts across many variables simultaneously.
>
> My project tackles this with multivariate SPC. I start by standardizing all 590 features and running PCA, which compresses them into about 5 uncorrelated components that explain at least 80% of the total variance. Then I compute two statistics on every wafer run: Hotelling's T² measures how far a run deviates from the normal process center within that compressed space, and SPE measures how much of the run can't be explained by the model at all — which catches a completely different class of anomaly.
>
> I set upper control limits using chi-squared approximations at 95% confidence and flag any run that exceeds either limit. The result is a single, unified anomaly score for each wafer run that reflects the entire 590-variable process state, not just individual sensors."

---

## 8 Likely Interview Questions with Answers

---

**Q1: Why use PCA before computing T² and SPE? Why not just compute T² directly on all 590 variables?**

> Computing T² directly on 590 variables requires inverting the full 590×590 covariance matrix, which is numerically unstable when variables are correlated (near-singular matrix) and requires far more data than observations to estimate reliably. PCA solves both problems: it produces orthogonal components (eigenvalues instead of a full matrix to invert) and concentrates the process information into a small number of well-estimated components. It also denoises the data by separating the structured variation from random sensor noise in the residual space — that residual is exactly what SPE monitors.

---

**Q2: What is the difference between Hotelling's T² and SPE? Can't one replace the other?**

> No — they monitor complementary things. T² measures deviations within the subspace defined by the retained PCs. If a process shifts in a direction already seen during normal operation (same direction, just farther out), T² catches it. SPE measures the residual — the part of the observation that the PCA model can't explain at all. If a completely new pattern of variation appears (a new failure mode, a different kind of excursion), it shows up in SPE but may be invisible in T². You need both statistics for complete monitoring coverage.

---

**Q3: How did you choose how many principal components to retain?**

> I used the cumulative explained variance criterion: retain the fewest components that together explain at least 80% of the total variance. This is a common practical threshold — it captures most of the meaningful process structure while excluding the noisy, low-variance components that would add degrees of freedom to T² without improving detection power. In practice, engineers also look at the scree plot inflection point and consider domain knowledge about how many independent process factors are likely operating.

---

**Q4: How did you set the upper control limits for T² and SPE?**

> For T²: The chi-squared distribution with k degrees of freedom (k = number of retained PCs) provides the theoretical UCL at a given confidence level. At 95% confidence, the UCL is the 95th percentile of χ²(k). This is an asymptotic result that holds well for large samples.
>
> For SPE: I used the Jackson-Mudholkar approximation, which fits a scaled chi-squared distribution to the empirical mean and variance of in-control SPE values. This is a standard industry approach because SPE doesn't have a simple closed-form distribution.

---

**Q5: What would you do differently if you had real fab data instead of synthetic data?**

> Several things. First, I'd do a proper Phase I / Phase II split — train the PCA model and compute control limits on a clean, in-control historical dataset (Phase I), then apply those fixed limits to new incoming runs (Phase II). Second, real data has non-stationarities: recipe changes, chamber cleaning cycles, preventive maintenance, and run-to-run drift. I'd investigate adaptive or windowed PCA approaches to handle these. Third, for root-cause analysis, I'd implement contribution plots that decompose a high T² or SPE value back to the original 590 variables so engineers know which sensors drove the alarm.

---

**Q6: What is a false positive in this context and how did you handle it?**

> A false positive is a normal wafer run that gets flagged as anomalous. By setting UCLs at 95% confidence, we expect about 5% of normal runs to exceed the limit by chance — that's the theoretical false positive rate, and it's a deliberate engineering tradeoff between sensitivity and specificity. In this project, the observed false positive rate is consistent with that 5% expectation. In a real fab, engineers choose the confidence level based on the cost of investigation versus the cost of a missed excursion.

---

**Q7: How does this relate to commercial fault detection and classification (FDC) systems used in fabs?**

> Commercial FDC systems like Applied Materials' FDC Suite, KLA's KLARITY, or PDF Solutions' Exensio use exactly these statistical foundations — PCA-based dimensionality reduction, T² and SPE monitoring, and threshold-based alarms — implemented at scale with real-time data streams from equipment sensors (via SEMI E10/E30 standards). My project implements the same mathematical core in a transparent, auditable Python pipeline. Understanding the underlying statistics makes it easier to configure and interpret commercial FDC tools, tune sensitivity thresholds, and investigate alarm root causes.

---

**Q8: What would you add to this project if you had more time?**

> Three things: (1) **Contribution plots** — decompose flagged T² and SPE values back to individual sensor contributions so an engineer can identify the root cause of each alarm, not just the fact of the alarm. (2) **Phase I / Phase II training split** — train the model on a held-out set of known-good runs and evaluate detection performance on a separate test set, which gives a more realistic assessment of out-of-sample performance. (3) **Rolling/adaptive control limits** — implement a sliding window or EWMA-updated PCA model to handle gradual process drift over time, which is critical in real manufacturing environments where process parameters shift slowly over thousands of wafer runs.

---

## Key Terms to Know Cold

| Term | One-line definition |
|------|---------------------|
| SPC | Statistical Process Control — using statistics to monitor and control manufacturing processes |
| Multivariate SPC | SPC applied jointly to multiple correlated variables |
| PCA | Dimensionality reduction by projecting data onto orthogonal directions of maximum variance |
| Principal Component | An orthogonal direction of variance in the data; linear combination of original features |
| Eigenvalue | Variance explained by a principal component |
| Score | Projection of an observation onto a principal component |
| Loading | Coefficient relating a principal component to original variables |
| Hotelling's T² | Multivariate distance metric in the retained PC subspace |
| SPE / Q statistic | Squared residual norm — variation outside the PCA model |
| UCL | Upper Control Limit — threshold above which an observation is flagged |
| False positive rate | Fraction of normal observations incorrectly flagged |
| FDC | Fault Detection and Classification — automated alarm systems used in semiconductor fabs |
