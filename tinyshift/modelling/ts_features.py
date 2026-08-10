# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Union, List
import numpy as np
import pandas as pd


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
