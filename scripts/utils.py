"""
utils.py
--------
Shared utility functions used across the analysis pipeline.
"""

import numpy as np
from scipy import stats


def chi2_threshold(alpha: float, dof: int) -> float:
    """
    Return the chi-squared upper-tail threshold for Hotelling's T² control limit.

    Parameters
    ----------
    alpha : float
        Significance level (e.g., 0.05 → 95% confidence).
    dof : int
        Degrees of freedom (number of retained principal components).

    Returns
    -------
    float
        Upper control limit (UCL).
    """
    return stats.chi2.ppf(1 - alpha, df=dof)


def spe_threshold(spe_values: np.ndarray, alpha: float = 0.05) -> float:
    """
    Estimate the SPE upper control limit using the chi-squared approximation
    (Jackson & Mudholkar, 1979).

    Parameters
    ----------
    spe_values : array-like
        SPE values computed from the calibration (normal) data.
    alpha : float
        Significance level.

    Returns
    -------
    float
        SPE upper control limit.
    """
    mu    = np.mean(spe_values)
    sigma = np.var(spe_values)
    dof   = 2.0 * (mu ** 2) / sigma
    scale = sigma / (2.0 * mu)
    return scale * stats.chi2.ppf(1 - alpha, df=dof)


def flag_anomalies(t2: np.ndarray, spe: np.ndarray,
                   t2_ucl: float, spe_ucl: float) -> np.ndarray:
    """
    Flag observations as anomalous if they exceed either T² or SPE UCL.

    Parameters
    ----------
    t2      : array of Hotelling's T² values.
    spe     : array of SPE values.
    t2_ucl  : T² upper control limit.
    spe_ucl : SPE upper control limit.

    Returns
    -------
    np.ndarray of bool
        True where observation is flagged as anomalous.
    """
    return (t2 > t2_ucl) | (spe > spe_ucl)


def print_summary(t2: np.ndarray, spe: np.ndarray,
                  t2_ucl: float, spe_ucl: float) -> None:
    """Print a concise anomaly detection summary to stdout."""
    flags = flag_anomalies(t2, spe, t2_ucl, spe_ucl)
    n_total   = len(flags)
    n_flagged = flags.sum()
    print("\n── Anomaly Detection Summary ──────────────────────────")
    print(f"   Total observations : {n_total}")
    print(f"   T²  UCL (@95%)     : {t2_ucl:.2f}")
    print(f"   SPE UCL (@95%)     : {spe_ucl:.4f}")
    print(f"   Flagged (T² only)  : {(t2 > t2_ucl).sum()}")
    print(f"   Flagged (SPE only) : {(spe > spe_ucl).sum()}")
    print(f"   Flagged (either)   : {n_flagged}  ({100*n_flagged/n_total:.1f}%)")
    print("───────────────────────────────────────────────────────\n")
