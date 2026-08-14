# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Union, Tuple, List, Dict, Any, Optional
import numpy as np
import scipy
import math
import pandas as pd
from statsmodels.tsa.seasonal import DecomposeResult
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from statsmodels.nonparametric.smoothers_lowess import lowess


def detrend(
    df: pd.DataFrame,
    frac: float = 0.2,
    robust: bool = True,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pd.DataFrame:
    """
    Decompose a Nixtla-format panel into LOWESS trend and residual.

    Parameters
    ----------
    df : pd.DataFrame
        Nixtla long-format panel with identifier, timestamp and target columns.
        Additional columns are preserved. Missing values in the target are linearly
        interpolated only while estimating the trend; original values are
        preserved in the output.
    frac : float, default=0.2
        Fraction of observations used for each local LOWESS regression.
        Larger values produce a smoother trend.
    robust : bool, default=True
        Whether to perform robustifying LOWESS iterations that down-weight
        outliers.
    id_col : str, default="unique_id"
        Column identifying each time series.
    time_col : str, default="ds"
        Column containing timestamps or integer time steps.
    target_col : str, default="y"
        Column containing the observed target values.
    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``trend`` and ``detrended`` appended. Rows retain
        their original order.

    Raises
    ------
    TypeError
        If ``df`` is not a pandas DataFrame.
    ValueError
        If a required Nixtla column is missing, a series has fewer than two
        observations, or ``frac`` is not in the interval ``(0, 1]``.
    """

    if not 0 < frac <= 1:
        raise ValueError("frac must be greater than 0 and less than or equal to 1.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame in Nixtla long format.")

    if not 0 < frac <= 1:
        raise ValueError("frac must be greater than 0 and less than or equal to 1.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame in Nixtla long format.")

    result = df.copy()

    counts = result.groupby(id_col, observed=True)[target_col].transform("count")
    if (counts < 2).any():
        raise ValueError(
            "Each unique_id series must contain at least two observations."
        )

    result = result.sort_values([id_col, time_col])

    clean_series = result.groupby(id_col, observed=True)[target_col].transform(
        lambda g: g.interpolate(method="linear", limit_direction="both")
    )

    it = 3 if robust else 0

    def _apply_lowess(s: pd.Series) -> pd.Series:
        """Applies LOWESS smoothing to a pandas Series."""
        y = s.to_numpy(dtype=float)
        x = np.arange(len(y))
        trend = lowess(y, x, frac=frac, it=it, return_sorted=False)
        return pd.Series(trend, index=s.index)

    result["trend"] = clean_series.groupby(result[id_col], observed=True).transform(
        _apply_lowess
    )

    result["detrended"] = result[target_col] - result["trend"]

    return result.loc[df.index]


def detect_seasonal_periods(
    series: Union[np.ndarray, List[float], pd.Series, pd.DataFrame],
    top_k: int = 3,
    noise_threshold_factor: float = 2.0,
    unique_id_col: str = "unique_id",
    target_col: Optional[str] = None,
) -> Union[List[int], Dict[Any, List[int]]]:
    """
    Identifies candidate seasonal periods in a time series using Fast Fourier Transform (FFT).

    The function detrends the series linearly to eliminate DC offset bias, calculates
    the Real FFT, identifies spectral magnitude peaks exceeding a noise threshold,
    and converts the dominant frequencies into discrete seasonal periods.

    Parameters
    ----------
    series : Union[np.ndarray, List[float], pd.Series, pd.DataFrame]
        Input 1D time series data or a panel DataFrame. A DataFrame must
        contain `unique_id_col` and a target column, and returns periods per ID.
    top_k : int, default=3
        Maximum number of candidate periods to return, ordered by spectral power.
    noise_threshold_factor : float, default=2.0
        Multiplier applied to the average spectral amplitude to filter out
        background noise peaks.
    unique_id_col : str, default="unique_id"
        DataFrame column containing the series identifiers.
    target_col : str, optional
        DataFrame column containing the values. If omitted, ``y`` is selected
        when available; otherwise the DataFrame must contain exactly one
        numeric column besides `unique_id_col`.

    Returns
    -------
    List[int] or dict
        For 1D input, a list of candidate seasonal periods sorted by dominant
        magnitude. For DataFrame input, a dictionary mapping each unique ID to
        its list of candidate periods. Empty lists indicate no significant
        periodic peaks.

    Raises
    ------
    ValueError
        If the input is invalid, the input length is less than 4, if `top_k` is
        non-positive, or if the DataFrame target column is ambiguous.
    """
    if top_k <= 0:
        raise ValueError(f"'top_k' must be a positive integer, got {top_k}.")

    if isinstance(series, pd.DataFrame):
        if unique_id_col not in series.columns:
            raise ValueError(
                f"DataFrame must contain the unique ID column {unique_id_col!r}."
            )

        if target_col is None:
            if "y" in series.columns:
                target_col = "y"
            else:
                numeric_columns = [
                    column
                    for column in series.columns
                    if column != unique_id_col
                    and pd.api.types.is_numeric_dtype(series[column])
                ]
                if len(numeric_columns) != 1:
                    raise ValueError(
                        "Could not infer the target column. Provide `target_col` "
                        "or include a single numeric column besides the ID."
                    )

                target_col = numeric_columns[0]
        elif target_col not in series.columns:
            raise ValueError(
                f"DataFrame does not contain target column {target_col!r}."
            )

        return {
            unique_id: detect_seasonal_periods(
                group[target_col],
                top_k=top_k,
                noise_threshold_factor=noise_threshold_factor,
            )
            for unique_id, group in series.groupby(unique_id_col, sort=False)
        }

    if isinstance(series, pd.Series):
        s = series.dropna().to_numpy(dtype=np.float64)
    else:
        s = np.asarray(series, dtype=np.float64)
        s = s[~np.isnan(s)]

    n = len(s)
    if n < 4:
        raise ValueError(f"Input series must have at least 4 observations, got {n}.")
    x = np.arange(n)
    poly_fit = np.polyfit(x, s, 1)
    signal = s - np.polyval(poly_fit, x)

    fft_vals = rfft(signal)
    freqs = rfftfreq(n, d=1.0)
    amplitudes = np.abs(fft_vals)

    non_dc_amplitudes = amplitudes[1:]
    if len(non_dc_amplitudes) == 0:
        return []

    noise_level = np.mean(non_dc_amplitudes) * noise_threshold_factor
    peaks, _ = find_peaks(non_dc_amplitudes, height=noise_level)
    peaks = peaks + 1

    if len(peaks) == 0:
        return []

    sorted_peaks = sorted(peaks, key=lambda idx: amplitudes[idx], reverse=True)[:top_k]

    periods = []
    for p in sorted_peaks:
        freq = freqs[p]
        if freq > 0:
            period = int(round(1.0 / freq))
            if 1 < period <= n // 2 and period not in periods:
                periods.append(period)

    return sorted(periods)


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


def extract_mstl_components(
    result: DecomposeResult, periods: Union[int, List[int]]
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

    period_list = [periods] if isinstance(periods, int) else list(periods)

    df = pd.DataFrame()
    df["data"] = result.observed
    df["trend"] = result.trend

    seasonal = np.asarray(result.seasonal)

    n_seasonal_channels = 1 if seasonal.ndim == 1 else seasonal.shape[1]
    if len(period_list) != n_seasonal_channels:
        raise ValueError(
            f"Number of provided periods ({len(period_list)}) does not match "
            f"the number of seasonal components in result ({n_seasonal_channels})."
        )

    if seasonal.ndim == 1:
        df[f"seasonal_{period_list[0]}"] = seasonal
    else:
        for i, p in enumerate(period_list):
            df[f"seasonal_{p}"] = seasonal[:, i]

    df["resid"] = result.resid
    return df
