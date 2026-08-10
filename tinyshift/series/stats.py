# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Union, Tuple, List
import numpy as np
import scipy
import math
import pandas as pd
from statsmodels.tsa.seasonal import DecomposeResult


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


def relative_strength_index(
    X: Union[np.ndarray, List[float]],
    rolling_window: int = 14,
) -> np.ndarray:
    """
    Feature transformer that computes the Relative Strength Index (RSI) for a given time series.

    The RSI is a momentum oscillator that quantifies the magnitude and direction of recent movements in a time series.
    Its values range from 0 to 100 and are commonly used to indicate different momentum regimes.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
        Time series data (e.g., closing prices).
    rolling_window : int, optional (default=14)
        The number of periods to use for calculating the RSI.

    Returns
    -------
    rsi : ndarray, shape (n_samples,)
        The RSI values for the time series.

    Notes
    -----
    - The RSI is calculated from the average gains and losses of returns over the specified rolling_window.
    - The first RSI value is computed after `rolling_window` periods.
    - Higher values indicate stronger positive momentum; lower values indicate stronger negative momentum.
    - Preserves the length of the input series; the first `rolling_window` values are initialized with the first computed RSI.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    deltas = np.diff(X)
    seed = deltas[: rolling_window + 1]
    mean_gain = seed[seed >= 0].sum() / rolling_window
    mean_loss = -seed[seed < 0].sum() / rolling_window
    rs = mean_gain / mean_loss if mean_loss != 0.0 else 0.0
    rsi = np.zeros_like(X)
    rsi[:rolling_window] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(rolling_window, len(X)):
        delta = deltas[i - 1]
        gain = max(delta, 0)
        loss = -min(delta, 0)
        mean_gain = (mean_gain * (rolling_window - 1) + gain) / rolling_window
        mean_loss = (mean_loss * (rolling_window - 1) + loss) / rolling_window
        rs = mean_gain / mean_loss if mean_loss != 0 else 0
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def standardize_returns(
    X: Union[np.ndarray, List[float]],
    log: bool = True,
    standardize: bool = True,
) -> np.ndarray:
    """
    Calculates and normalizes the returns of a time series.

    The function computes either logarithmic or simple returns from the
    input series and then standardizes the resulting return series
    (Z-score normalization).

    Parameters
    ----------
    X : array-like
        A 1-dimensional time series (e.g., prices, sales figures, volume).
    log : bool, default=True
        If True, calculates **logarithmic returns**: r_t = ln(X_t / X_{t-1}).
        If False, calculates **simple (percentage) returns**: R_t = (X_t / X_{t-1}) - 1.
    standardize : bool, default=True
        If True, standardizes the return series to have zero mean and unit variance.

    Returns
    -------
    norm : np.ndarray
        The normalized return series (with zero mean and unit standard deviation).

    Raises
    ------
    ValueError
        If the input data 'X' is not 1-dimensional.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    ratios = X[1:] / X[:-1]
    returns = np.log(ratios) if log else ratios - 1
    returns = (returns - np.mean(returns)) / np.std(returns) if standardize else returns
    return np.concatenate([[np.nan], returns])


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


def extract_mstl_components(
    result: DecomposeResult, period_list: List[int]
) -> pd.DataFrame:
    """
    Transforms a statsmodels MSTL decomposition result into a structured DataFrame.

    Extracts observed data, trend, seasonal component(s), and residual signals
    from a fitted `DecomposeResult` instance and maps them into explicit,
    identifiable column names.

    Parameters
    ----------
    result : DecomposeResult
        Fitted decomposition output object obtained from calling `.fit()`
        on a `statsmodels.tsa.seasonal.MSTL` instance.
    periods : int or list of int
        Seasonal period(s) used during the MSTL fit procedure. Used to dynamically
        suffix seasonal column names (e.g., `seasonal_7`, `seasonal_365`). Must match
        the order and number of periods passed to the original MSTL estimator.

    Returns
    -------
    components_df : pd.DataFrame
        Structured DataFrame containing the unnested decomposition components:
        - ``data`` : Original observed time series values.
        - ``trend`` : Extracted smoothed trend component.
        - ``seasonal_<period>`` : Individual seasonal component for each specified period.
          If a single period is provided, the column is named ``seasonal_<period>``.
        - ``resid`` : Remaining unexplained residual noise component.

    Raises
    ------
    ValueError
        If the number of provided periods does not match the number of seasonal
        channels present in the `DecomposeResult` instance.
    TypeError
        If `result` is not an instance of `DecomposeResult`.
    """

    if not isinstance(result, DecomposeResult):
        raise TypeError(
            f"Expected 'result' to be a statsmodels DecomposeResult, got {type(result).__name__}."
        )
    df = pd.DataFrame()
    df["data"] = result.observed
    df["trend"] = result.trend

    seasonal = np.asarray(result.seasonal)
    if seasonal.ndim == 1:
        df[f"seasonal_{period_list[0]}"] = seasonal
    else:
        for i, p in enumerate(period_list):
            df[f"seasonal_{p}"] = seasonal[:, i]

    df["resid"] = result.resid
    return df


def fourier_seasonality(
    df: pd.DataFrame,
    time_col: str,
    seasonality: List[str],
):
    """
    Adds Fourier-based seasonal features to the dataframe.

    Parameters
    -----------
    df : pandas.DataFrame
        Input dataframe with time column
    time_col : str
        Name of the datetime column
    seasonalities : list, optional
        List of seasonalities to include. Options:
        ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']
        Default: ['weekly', 'yearly']

    Returns
    --------
    pandas.DataFrame
        DataFrame with added Fourier seasonal features

    """
    df = df.copy()

    seasonality_config = {
        "daily": {"period": 24, "value_func": lambda dt: dt.hour, "name": "daily"},
        "weekly": {
            "period": 7,
            "value_func": lambda dt: dt.dayofweek,
            "name": "weekly",
        },
        "monthly": {"period": 12, "value_func": lambda dt: dt.month, "name": "monthly"},
        "quarterly": {
            "period": 4,
            "value_func": lambda dt: dt.quarter,
            "name": "quarterly",
        },
        "yearly": {
            "period": 365,
            "value_func": lambda dt: dt.dayofyear,
            "name": "yearly",
        },
    }

    for season in seasonality:
        if season not in seasonality_config:
            raise ValueError(
                f"Unknown seasonality: {season}. "
                f"Available options: {list(seasonality_config.keys())}"
            )

        config = seasonality_config[season]
        period = config["period"]
        values = config["value_func"](df[time_col].dt)
        name = config["name"]

        df[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * values / period)

    return df


def estimate_history_length(seasonal_period: int, horizon: int) -> int:
    """
    Estimates a heuristic lag value (history window size) based on the seasonal
    period and the forecast horizon.

    This heuristic is commonly used in time series modeling
    to ensure the model's regressor includes enough historical data to capture
    the full seasonal cycle and the entire prediction range.

    The calculation follows the rule-of-thumb: L = 1.25 * max(S, H).

    Parameters
    ----------
    seasonal_period : int
        The known seasonal period (S) of the time series (e.g., 7 for weekly data, 365 for daily/yearly data).

    horizon : int
        The desired forecast horizon (H) in the same units as the seasonal period.

    Returns
    -------
    int
        The suggested historical lag value (L). It is implicitly an integer as lag values are typically discrete.
    """

    return int(1.25 * np.max([seasonal_period, horizon]))
