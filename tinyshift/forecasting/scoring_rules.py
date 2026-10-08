"""Proper scoring rules for probabilistic forecasts."""

import numpy as np
from numpy.polynomial.legendre import leggauss


def _finite_vector(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def crps_quantile(
    y_true: np.ndarray,
    quantiles: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Approximate row-wise CRPS from predictive quantiles."""
    y_true = _finite_vector(y_true, "y_true")
    quantiles = np.asarray(quantiles, dtype=float)
    probabilities = _finite_vector(probabilities, "probabilities")
    if probabilities.size < 2:
        raise ValueError("probabilities must contain at least two values.")
    if quantiles.shape != (y_true.size, probabilities.size):
        raise ValueError(
            "quantiles must have shape (len(y_true), len(probabilities))."
        )
    if not np.all(np.isfinite(quantiles)):
        raise ValueError("quantiles must contain only finite values.")
    if np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
        raise ValueError("probabilities must be strictly between 0 and 1.")
    if np.any(np.diff(probabilities) <= 0.0):
        raise ValueError("probabilities must be strictly increasing.")

    errors = y_true[:, None] - quantiles
    loss = np.where(
        errors >= 0.0,
        probabilities * errors,
        (probabilities - 1.0) * errors,
    )
    return 2.0 * np.trapezoid(loss, probabilities, axis=1)


def crps_ensemble(y_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Compute row-wise CRPS for empirical predictive samples.

    ``samples`` has shape ``(n_observations, n_samples)``. The implementation
    uses the exact empirical-distribution identity in O(M log M) per row.
    """
    y_true = _finite_vector(y_true, "y_true")
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] != y_true.size:
        raise ValueError("samples must have shape (len(y_true), n_samples).")
    if samples.shape[1] < 1:
        raise ValueError("samples must contain at least one member per observation.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")

    sorted_samples = np.sort(samples, axis=1)
    n_samples = samples.shape[1]
    coefficients = 2 * np.arange(1, n_samples + 1) - n_samples - 1
    pairwise_term = np.sum(sorted_samples * coefficients, axis=1) / n_samples**2
    observation_term = np.mean(np.abs(samples - y_true[:, None]), axis=1)
    return observation_term - pairwise_term


def crps_distribution(
    y_true: np.ndarray,
    distribution,
    n_nodes: int = 100,
) -> np.ndarray:
    """Approximate row-wise CRPS from a distribution exposing ``ppf``."""
    y_true = _finite_vector(y_true, "y_true")
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int) or n_nodes < 2:
        raise ValueError("n_nodes must be an integer greater than or equal to 2.")
    if not callable(getattr(distribution, "ppf", None)):
        raise TypeError("distribution must expose a callable ppf method.")

    nodes, weights = leggauss(n_nodes)
    probabilities = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    quantiles = np.asarray(distribution.ppf(probabilities), dtype=float)
    expected_shape = (y_true.size, probabilities.size)
    if quantiles.shape != expected_shape:
        raise ValueError("distribution quantiles are not aligned with y_true.")
    if not np.all(np.isfinite(quantiles)):
        raise ValueError("distribution quantiles must contain only finite values.")

    errors = y_true[:, None] - quantiles
    loss = np.where(
        errors >= 0.0,
        probabilities * errors,
        (probabilities - 1.0) * errors,
    )
    return 2.0 * np.sum(loss * weights, axis=1)


def ncrps(crps_values: np.ndarray | float, scale: np.ndarray | float) -> np.ndarray:
    """Normalize CRPS values by a positive target scale."""
    crps_values = np.asarray(crps_values, dtype=float)
    scale = np.asarray(scale, dtype=float)
    try:
        crps_values, scale = np.broadcast_arrays(crps_values, scale)
    except ValueError as exc:
        raise ValueError("crps_values and scale must be broadcast-compatible.") from exc
    if not np.all(np.isfinite(crps_values)) or np.any(crps_values < 0.0):
        raise ValueError("crps_values must contain finite non-negative values.")
    result = np.full(crps_values.shape, np.nan, dtype=float)
    valid_scale = np.isfinite(scale) & (scale > 0.0)
    np.divide(crps_values, scale, out=result, where=valid_scale)
    return result


def mwis(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """Compute the mean Winkler interval score for a central interval."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    valid = ~(np.isnan(y_true) | np.isnan(lower) | np.isnan(upper))
    if not valid.any():
        return np.nan

    y_true, lower, upper = y_true[valid], lower[valid], upper[valid]
    width = upper - lower
    penalty_lower = (2.0 / alpha) * (lower - y_true) * (y_true < lower)
    penalty_upper = (2.0 / alpha) * (y_true - upper) * (y_true > upper)
    return float(np.mean(width + penalty_lower + penalty_upper))
