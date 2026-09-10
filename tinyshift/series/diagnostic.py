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
    Test the significance of a harmonic component at a candidate period.

    A harmonic regression is fitted using sine and cosine terms at the
    candidate period:

        y(t) = intercept
             + beta_sin * sin(2*pi*t / period)
             + beta_cos * cos(2*pi*t / period)
             + error

    The null hypothesis is that the harmonic component does not explain
    significant variation in the series:

        H0: beta_sin = beta_cos = 0

    The hypothesis is evaluated with an F-test comparing the variation
    explained by the harmonic regression with the residual variation.

    Parameters
    ----------
    y_detrended : array-like
        One-dimensional detrended series.
    period : int
        Candidate seasonal period in observations. Must be greater than 1.

    Returns
    -------
    f_statistic : float
        F statistic for the joint significance of the sine and cosine terms.
    p_value : float
        P-value associated with the F statistic.
    """
    y_detrended = np.asarray(y_detrended, dtype=np.float64)

    if y_detrended.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if not np.isfinite(y_detrended).all():
        raise ValueError("Input data must contain only finite values")
    if isinstance(period, bool) or not isinstance(period, int) or period <= 1:
        raise ValueError("'period' must be an integer greater than 1")

    # Build the harmonic regression.
    n_observations = len(y_detrended)
    time_index = np.arange(n_observations)

    sine_component = np.sin(2 * np.pi * time_index / period)
    cosine_component = np.cos(2 * np.pi * time_index / period)

    design_matrix = np.column_stack(
        [
            np.ones(n_observations),
            sine_component,
            cosine_component,
        ]
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        design_matrix,
        y_detrended,
        rcond=None,
    )

    predicted_values = design_matrix @ coefficients

    # Decompose the total variation into explained and residual variation.
    total_sum_squares = np.sum((y_detrended - np.mean(y_detrended)) ** 2)
    residual_sum_squares = np.sum((y_detrended - predicted_values) ** 2)
    explained_sum_squares = max(
        0.0,
        total_sum_squares - residual_sum_squares,
    )

    model_rank = np.linalg.matrix_rank(design_matrix)
    harmonic_degrees_freedom = model_rank - 1
    residual_degrees_freedom = n_observations - model_rank

    if (
        harmonic_degrees_freedom <= 0
        or residual_degrees_freedom <= 0
        or residual_sum_squares <= 0
    ):
        return 0.0, 1.0

    explained_mean_square = explained_sum_squares / harmonic_degrees_freedom
    residual_mean_square = residual_sum_squares / residual_degrees_freedom

    f_statistic = explained_mean_square / residual_mean_square

    p_value = scipy.stats.f.sf(
        f_statistic,
        harmonic_degrees_freedom,
        residual_degrees_freedom,
    )

    return float(f_statistic), float(p_value)
