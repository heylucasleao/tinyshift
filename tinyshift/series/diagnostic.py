# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import scipy


def variance_ratio(
    X: Union[np.ndarray, List[float]],
    horizon: int = 2,
) -> Tuple[float, float, float]:
    """
    Performs the Lo-MacKinlay variance ratio test for serial dependence.

    The variance ratio compares the variance of changes over a horizon `k`
    with `k` times the variance of one-period changes.

    Under the random-walk null hypothesis:

        VR(k) = 1

    Values greater than 1 indicate positive serial dependence
    (persistence), while values below 1 indicate negative serial dependence
    (mean reversion / anti-persistence).

    Parameters
    ----------
    X: Union[np.ndarray, List[float]]
        One-dimensional time series in level form.

    horizon: int, default=2
        Aggregation horizon `k`. Must be greater than 1 and smaller than
        the number of one-period increments.

    Returns
    -------
    Tuple[float, float, float]
        (variance_ratio, z_statistic, p_value)

        variance_ratio: float
            Lo-MacKinlay variance ratio estimate.

            - VR > 1: positive serial dependence / persistence
            - VR = 1: behavior consistent with a random walk
            - VR < 1: negative serial dependence / mean reversion

        z_statistic: float
            Lo-MacKinlay homoscedastic test statistic for H0: VR(k) = 1.

        p_value : float
            Two-sided p-value for the random-walk null hypothesis.

    Raises
    ------
    ValueError
        If the input is not one-dimensional, contains insufficient data,
        or if `horizon` is invalid.

    Notes
    -----
    The implementation uses overlapping k-period changes and the
    finite-sample variance estimator proposed by Lo and MacKinlay.

    The null hypothesis is:

        H0: VR(k) = 1

    which corresponds to uncorrelated one-period increments.

    The test statistic assumes homoscedastic increments.
    """

    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    if len(X) < 30:
        raise ValueError("Insufficient data points (minimum 30 required)")

    if horizon <= 1:
        raise ValueError("'horizon' must be greater than 1")

    increments = np.diff(X)
    n = len(increments)

    if horizon >= n:
        raise ValueError("'horizon' must be smaller than the number of increments")

    mean_increment = np.mean(increments)

    # One-period variance estimator.
    variance_1 = np.sum((increments - mean_increment) ** 2) / (n - 1)

    if variance_1 <= np.finfo(np.float64).eps:
        raise ValueError(
            "Variance ratio is undefined when one-period "
            "increments have zero variance"
        )

    # Overlapping k-period changes.
    k_period_changes = X[horizon:] - X[:-horizon]

    # Finite-sample correction from Lo-MacKinlay.
    m = horizon * (n - horizon + 1) * (1.0 - horizon / n)

    variance_k = np.sum((k_period_changes - horizon * mean_increment) ** 2) / m

    ratio = variance_k / variance_1

    # Homoscedastic asymptotic variance of VR(k).
    phi = 2.0 * (2.0 * horizon - 1.0) * (horizon - 1.0) / (3.0 * horizon * n)

    z_statistic = (ratio - 1.0) / np.sqrt(phi)

    p_value = 2.0 * scipy.stats.norm.sf(abs(z_statistic))

    return (
        float(ratio),
        float(z_statistic),
        float(p_value),
    )


def trend_significance(
    X: Union[np.ndarray, List[float]],
) -> Tuple[float, float]:
    """
    Performs a linear regression against time (index) to check for a significant
    linear trend in the input data.

    The function calculates the R-squared value and the p-value of the
    hypothesis test where the null hypothesis is that the slope of the
    regression line is zero (i.e., no linear trend).

    Parameters
    ----------
    X : Union[np.ndarray, List[float]]
        One-dimensional array or time series data (e.g., a numpy array or list).

    Returns
    -------
    Tuple[float, float]
        (R-squared, p-value)
        r_squared : float
            The coefficient of determination (R²), representing the proportion
            of variance in the data explained by the linear trend.
        p_value : float
            The two-sided p-value for a hypothesis test whose null hypothesis is
            that the slope of the regression line is zero.

    Raises
    ------
    ValueError
        If the input data is not 1-dimensional.

    Notes
    -----
    A 'significant' linear trend for detrending purposes is typically considered
    when:
    1. R² is high enough (e.g., > 0.1), suggesting a non-trivial variance
       explained.
    2. p-value is low enough (e.g., < 0.05), indicating the slope is
       statistically different from zero.

    The initial criteria described in the code comments are:
    - R² > 0.1 (10% of variance explained)
    - p-value < 0.05 (statistically significant trend)
    """

    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    time_index = np.arange(len(X))
    _, _, r_value, p_value, _ = scipy.stats.linregress(time_index, X)
    r_squared = r_value**2

    return r_squared, p_value


def seasonal_strength(
    seasonal_component: Union[np.ndarray, List[float], pd.Series],
    residuals: Union[np.ndarray, List[float], pd.Series],
) -> float:
    """Calculate Hyndman's seasonal-strength measure from decomposition components."""
    seasonal_component = np.asarray(seasonal_component, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)

    var_resid = np.var(residuals, ddof=1)
    var_seas_resid = np.var(seasonal_component + residuals, ddof=1)
    return float(
        max(0.0, 1.0 - (var_resid / var_seas_resid)) if var_seas_resid > 0 else 0.0
    )


def harmonic_significance(
    y_detrended: Union[np.ndarray, List[float], pd.Series],
    period: int,
) -> Tuple[float, float]:
    """Test sinusoidal terms for a candidate period using an F-test."""
    y_detrended = np.asarray(y_detrended, dtype=np.float64)
    if y_detrended.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if not np.isfinite(y_detrended).all():
        raise ValueError("Input data must contain only finite values")
    if isinstance(period, bool) or not isinstance(period, int) or period <= 1:
        raise ValueError("'period' must be an integer greater than 1")

    n = len(y_detrended)
    t = np.arange(n)
    sin_t = np.sin(2 * np.pi * t / period)
    cos_t = np.cos(2 * np.pi * t / period)
    X_design = np.column_stack([np.ones(n), sin_t, cos_t])

    beta, _, _, _ = np.linalg.lstsq(X_design, y_detrended, rcond=None)
    y_pred = X_design @ beta
    ss_tot = np.sum((y_detrended - np.mean(y_detrended)) ** 2)
    ss_res = np.sum((y_detrended - y_pred) ** 2)
    ss_reg = ss_tot - ss_res

    model_rank = np.linalg.matrix_rank(X_design)
    df_reg = model_rank - 1
    df_res = n - model_rank

    if df_reg > 0 and df_res > 0 and ss_res > 0:
        f_stat = (ss_reg / df_reg) / (ss_res / df_res)
        p_val = scipy.stats.f.sf(f_stat, df_reg, df_res)
    else:
        f_stat, p_val = 0.0, 1.0

    return float(f_stat), float(p_val)
