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
    Calculate the Lo-MacKinlay variance ratio test for serial dependence.

    The variance ratio compares the variance of changes over a horizon ``k``
    with ``k`` times the variance of one-period changes. Under the random-walk
    null hypothesis, the statistic is approximately one.

    Parameters
    ----------
    X : Union[np.ndarray, List[float]]
        One-dimensional time series in level form.
    horizon : int, default=2
        Aggregation horizon ``k``. Must be greater than 1 and smaller than the
        number of one-period increments.

    Returns
    -------
    Tuple[float, float, float]
        (variance_ratio, z_statistic, p_value)
        variance_ratio : float
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
    Values greater than 1 indicate positive serial dependence (persistence),
    while values below 1 indicate negative serial dependence (mean reversion or
    anti-persistence). The implementation uses overlapping ``k``-period changes
    and the finite-sample variance correction proposed by Lo and MacKinlay.

    References
    ----------
    Lo, A. W., & MacKinlay, A. C. (1988). Stock market prices do not follow
    random walks: Evidence from a simple specification test. Review of Financial
    Studies, 1(1), 41-66.
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
) -> Tuple[float, float, float]:
    """
    Test whether a linear trend is statistically significant.

    The function fits a least-squares line against the observation index and
    evaluates whether the slope differs from zero. The result includes the slope,
    the coefficient of determination, and the two-sided p-value.

    Parameters
    ----------
    X : Union[np.ndarray, List[float]]
        One-dimensional time series or sequence of observations.

    Returns
    -------
    Tuple[float, float, float]
        (slope, r_squared, p_value)
        slope : float
            Linear change in the target per observation.
        r_squared : float
            Proportion of variance explained by the fitted linear trend.
        p_value : float
            Two-sided p-value for the null hypothesis that the slope is zero.

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
    slope, _, r_value, p_value, _ = scipy.stats.linregress(time_index, X)
    r_squared = r_value**2

    return float(slope), float(r_squared), float(p_value)


def harmonic_significance(
    y_detrended: Union[np.ndarray, List[float], pd.Series],
    period: int,
) -> Tuple[float, float]:
    """
    Test whether a sinusoidal component at a candidate period is significant.

    The function regresses the detrended series on sine and cosine terms for the
    supplied period and returns the resulting F statistic and p-value.

    Parameters
    ----------
    y_detrended : Union[np.ndarray, List[float], pd.Series]
        One-dimensional detrended series.
    period : int
        Candidate seasonal period in observations. Must be greater than 1.

    Returns
    -------
    Tuple[float, float]
        (f_statistic, p_value)
        f_statistic : float
            F statistic for the harmonic regression.
        p_value : float
            p-value associated with the null hypothesis that the seasonal term is
            not significant.

    Notes
    -----
    This diagnostic is used to assess whether an apparent cycle at a given period
    is statistically meaningful beyond a generic harmonic fluctuation.
    """
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
