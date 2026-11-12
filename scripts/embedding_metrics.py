"""Descriptive item-level diagnostics for the proposal's embedding question."""

import numpy as np
from scipy.stats import pearsonr, spearmanr


def correlation(left, right, method="spearman"):
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 1 or right.shape != left.shape or len(left) < 2:
        raise ValueError("Correlations require equally sized one-dimensional arrays")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Non-finite diagnostic values")
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    if method == "spearman":
        return float(spearmanr(left, right).statistic)
    if method == "pearson":
        return float(pearsonr(left, right).statistic)
    raise ValueError("Unknown correlation method")


def embedding_diagnostics(degree, raw_norm, combined_norm, last_norm, exposure):
    sqrt_degree = np.sqrt(degree)
    return {
        "rho_degree_exposure": correlation(degree, exposure),
        "rho_raw_norm_sqrt_degree": correlation(raw_norm, sqrt_degree),
        "rho_combined_norm_sqrt_degree": correlation(combined_norm, sqrt_degree),
        "rho_last_norm_sqrt_degree": correlation(last_norm, sqrt_degree),
        "rho_combined_norm_exposure": correlation(combined_norm, exposure),
        "pearson_combined_norm_sqrt_degree": correlation(combined_norm, sqrt_degree, "pearson"),
    }
