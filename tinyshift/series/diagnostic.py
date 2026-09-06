# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import math
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import scipy

from .decomposition import detrend, extract_mstl_components


def hurst_exponent(
    X: Union[np.ndarray, List[float]],
    d: int = 1,
) -> Tuple[float, float]:
    """
    Calculate the Hurst exponent using a rescaled range (R/S) analysis approach with p-value for random walk hypothesis.

    The Hurst exponent is a measure of long-term memory of time series. It relates
    to the autocorrelations of the time series and the rate at which these decrease
    as the lag between pairs of values increases.

    Parameters
    ----------
    X : Union[np.ndarray, List[float]]
        Input 1D time series data for which to calculate the Hurst exponent.
        Must contain at least 30 samples.
    d : int, default=1
        The order of differencing to apply to the time series before analysis.
        Can be 0 (no differencing), 1 (first difference), or 2 (second difference).

    Returns
    -------
    Tuple[float, float]
        (Hurst exponent, p-value for H=0.5 hypothesis)
        The estimated Hurst exponent value. Interpretation:
        - 0 < H < 0.5: Mean-reverting (anti-persistent) series
        - H = 0.5: Geometric Brownian motion (random walk)
        - 0.5 < H < 1: Trending (persistent) series with long-term memory
        - H = 1: Perfectly trending series
        p-value interpretation:
        - p < threshold: Reject random walk hypothesis (significant persistence/mean-reversion)
        - p >= threshold: Cannot reject random walk hypothesis

    Raises
    ------
    ValueError
        If input data has less than 30 samples (insufficient for reliable estimation).
    TypeError
        If input is not a list or numpy array.

    Notes
    -----
    - The method uses differencing of order `d` to remove trends/non-stationarities.
    - The R/S analysis is performed over multiple window sizes to estimate the Hurst exponent.
    - A hypothesis test is conducted to assess if the estimated Hurst exponent significantly differs from 0.5 (random walk).
    """
    if d not in [0, 1, 2]:
        raise ValueError("Differencing order 'd' must be either 0, 1, or 2")

    X = np.asarray(X, dtype=np.float64)
    deltas = np.diff(X, n=d)
    size = len(deltas)

    if 30 > len(X):
        raise ValueError("Insufficient data points (minimum 30 required)")

    def _calculate_rescaled_ranges(
        deltas: np.ndarray, window_sizes: List[int]
    ) -> np.ndarray:
        """Helper function to calculate rescaled ranges (R/S) for each window size."""
        r_s = np.zeros(len(window_sizes), dtype=np.float64)

        for i, window_size in enumerate(window_sizes):
            n_windows = len(deltas) // window_size
            truncated_size = n_windows * window_size

            windows = deltas[:truncated_size].reshape(n_windows, window_size)

            means = np.mean(windows, axis=1, keepdims=True)
            std_devs = np.std(windows, axis=1, ddof=1)
            demeaned = windows - means
            cumulative_sums = np.cumsum(demeaned, axis=1)
            ranges = np.max(cumulative_sums, axis=1) - np.min(cumulative_sums, axis=1)

            r_s[i] = np.mean(ranges / std_devs)

        return r_s

    def _hypothesis_test_random_walk(hurst: float, se: float, n: int) -> float:
        """Helper function to test if Hurst exponent is significantly different from random_walk (0.5)"""
        random_walk = 0.5
        t_stat = (hurst - random_walk) / se
        ddof = n - 2
        return 2 * scipy.stats.t.sf(abs(t_stat), ddof)

    max_power = int(np.floor(math.log2(size)))
    window_sizes = [2**power for power in range(1, max_power + 1)]

    rescaled_ranges = _calculate_rescaled_ranges(deltas, window_sizes)

    log_sizes = np.log(window_sizes)
    log_r_s = np.log(rescaled_ranges)
    slope, _, _, _, se = scipy.stats.linregress(log_sizes, log_r_s)

    p_value = _hypothesis_test_random_walk(slope, se, len(window_sizes))

    return float(slope), float(p_value)


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


def seasonal_significance(
    y_detrended: Union[np.ndarray, List[float], pd.Series],
    seasonal_component: Union[np.ndarray, List[float], pd.Series],
    residuals: Union[np.ndarray, List[float], pd.Series],
    period: int,
) -> Tuple[float, float, float]:
    """
    Calculates seasonal strength (Hyndman's metric) and performs an F-test
    for seasonal significance using harmonic regression terms.

    Parameters
    ----------
    y_detrended : Union[np.ndarray, List[float], pd.Series]
        The time series data after trend removal.
    seasonal_component : Union[np.ndarray, List[float], pd.Series]
        The extracted seasonal component for the given period.
    residuals : Union[np.ndarray, List[float], pd.Series]
        The residual component from the decomposition.
    period : int
        The length of the seasonal cycle (e.g., 7 for weekly, 12 for monthly).

    Returns
    -------
    Tuple[float, float, float]
        (strength, f_stat, p_value)
        strength : float
            Seasonal strength index ranging from 0 to 1.
        f_stat : float
            F-statistic testing the joint significance of harmonic terms.
        p_value : float
            p-value corresponding to the F-test.
    """
    y_detrended = np.asarray(y_detrended, dtype=np.float64)
    seasonal_component = np.asarray(seasonal_component, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)

    var_resid = np.var(residuals, ddof=1)
    var_seas_resid = np.var(seasonal_component + residuals, ddof=1)
    strength = (
        max(0.0, 1.0 - (var_resid / var_seas_resid)) if var_seas_resid > 0 else 0.0
    )

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

    df_reg = 2
    df_res = n - 3

    if df_res > 0 and ss_res > 0:
        f_stat = (ss_reg / df_reg) / (ss_res / df_res)
        p_val = scipy.stats.f.sf(f_stat, df_reg, df_res)
    else:
        f_stat, p_val = 0.0, 1.0

    return float(strength), float(f_stat), float(p_val)
