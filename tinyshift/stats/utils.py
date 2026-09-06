# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Callable, List, Union

import numpy as np
import pandas as pd


def chebyshev_guaranteed_percentage(
    X: Union[np.ndarray, List[float]], interval: Union[np.ndarray, List[float]]
) -> float:
    """
    Computes the minimum percentage of data within a given interval using Chebyshev's inequality.

    Chebyshev's theorem guarantees that for any distribution, at least (1 - 1/k²) of the data lies
    within 'k' standard deviations from the mean. The coefficient 'k' is computed for each bound
    (lower and upper) independently, and the conservative (smaller) value is chosen to ensure a
    valid lower bound.

    Parameters:
    ----------
    X : array-like
        Input numerical data.
    interval : tuple (lower, upper)
        The interval of interest (lower and upper bounds). Use None for unbounded sides.

    Returns:
    -------
    float
        The minimum fraction (between 0 and 1) of data within the interval.
        Returns 0 if the interval is too wide (k ≤ 1), where the theorem provides no meaningful bound.

    Notes:
    -----
    - If `lower` is None, the interval is unbounded on the left.
    - If `upper` is None, the interval is unbounded on the right.
    """

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 1 or X.size < 2:
        raise ValueError("X must be a one-dimensional array with at least two values.")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values.")
    if not isinstance(interval, (tuple, list, np.ndarray)) or len(interval) != 2:
        raise ValueError("interval must contain exactly two bounds.")

    lower, upper = interval
    if lower is None and upper is None:
        return 1.0
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("The lower bound cannot exceed the upper bound.")

    mu = np.mean(X)
    std = np.std(X, ddof=1)
    if std == 0:
        inside = (lower is None or lower <= mu) and (upper is None or mu <= upper)
        return 1.0 if inside else 0.0
    k_values = []
    if lower is not None:
        k_lower = (mu - lower) / std
        k_values.append(k_lower)
    if upper is not None:
        k_upper = (upper - mu) / std
        k_values.append(k_upper)
    k = float(min(k_values))
    return 1 - (1 / (k**2)) if k > 1 else 0


def jackknife(
    X: Union[np.ndarray, List[float]],
    func: Callable = None,
    **kwargs,
) -> np.ndarray:
    """
    Apply a function using the jackknife approach on a 1D time series.
    Parameters
    ----------
    X : array-like, shape (n_samples,)
        1D time series data (e.g., log-prices).
    func : Callable
        Function to apply to each jackknife sample. Must accept a 1D array as first argument.
    **kwargs
        Additional keyword arguments to pass to `func`.
    Returns
    -------
    result : ndarray, shape (n_samples,)
        Array of function values for each jackknife sample.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if X.size < 2:
        raise ValueError("Input data must contain at least two values")
    if not callable(func):
        raise TypeError("func must be callable")

    result = np.array([func(np.delete(X, i), **kwargs) for i in range(X.shape[0])])

    return result


# Backwards-compatible alias for the original misspelled public name.
jacknife = jackknife


def mad(x):
    """
    Calculate the Median Absolute Deviation (MAD) of a 1D array.

    The MAD is a robust measure of variability that is less sensitive to outliers
    than the standard deviation. It is defined as the median of the absolute
    deviations from the data's median.

    Parameters
    ----------
    x : array-like
        Input data array.

    Returns
    -------
    float
        The median absolute deviation of the input data.
    """
    return np.median(np.absolute(x - np.median(x)))


def generate_lag(
    X: Union[np.ndarray, List[float]],
    lag=1,
):
    """
    Generate lagged differences for a 1D time series.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
        1D time series data.
    lag : int, optional (default=1)
        The lag interval.

    Returns
    -------
    result : ndarray, shape (n_samples,)
        Array containing NaNs for the first 'lag' elements followed by the lagged differences.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input array must be one-dimensional.")

    if (
        isinstance(lag, (bool, np.bool_))
        or not isinstance(lag, (int, np.integer))
        or lag <= 0
    ):
        raise ValueError("lag must be a positive integer.")
    if lag > len(X):
        raise ValueError("lag cannot be larger than the length of X.")

    return np.concatenate((np.nan * np.ones(lag), (X[lag:] - X[:-lag])))


def generate_panel_lags(
    df: pd.DataFrame,
    nlags: List[int],
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pd.DataFrame:
    """
    Generate lagged values and lagged differences for a Nixtla-style panel.

    Parameters
    ----------
    df : pd.DataFrame
        Nixtla-format panel containing identifier, timestamp, and target columns.
    nlags : list of int
        Lag intervals to generate.
    id_col : str, default="unique_id"
        Column identifying each time series.
    time_col : str, default="ds"
        Column containing timestamps or time steps.
    target_col : str, default="y"
        Column containing target values.

    Returns
    -------
    pd.DataFrame
        Copy of the input sorted by identifier and time, with columns named
        ``{target_col}_lag_{lag}`` and ``{target_col}_diff_lag_{lag}``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required = [id_col, time_col, target_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}.")

    if not nlags:
        raise ValueError("nlags must contain at least one lag.")

    if any(
        not isinstance(lag, int) or isinstance(lag, bool) or lag <= 0 for lag in nlags
    ):
        raise ValueError("All lags must be positive integers.")

    if len(set(nlags)) != len(nlags):
        raise ValueError("nlags cannot contain duplicates.")

    result = df.sort_values([id_col, time_col]).copy()
    grouped_target = result.groupby(id_col, sort=False)[target_col]

    for lag in nlags:
        result[f"{target_col}_lag_{lag}"] = grouped_target.shift(lag)
        result[f"{target_col}_diff_lag_{lag}"] = grouped_target.transform(
            lambda series, lag=lag: generate_lag(
                series.to_numpy(),
                lag=lag,
            )
        )

    return result


def remove_leading_zeros(group):
    """
    Removes leading zeros from a time series group.

    Parameters
    ----------
    group : pandas.DataFrame
        DataFrame containing time series data with a 'y' column.

    Returns
    -------
    pandas.DataFrame
        DataFrame with leading zeros removed, starting from the first non-zero value.
    """
    non_zero = group["y"].ne(0)
    if not non_zero.any():
        return group.iloc[0:0]
    first_non_zero_index = non_zero.idxmax()
    return group.loc[first_non_zero_index:]


def is_obsolete(group, days_obsoletes):
    """
    Determine if a time series group is obsolete based on recent data.

    Parameters
    ----------
    group : pandas.DataFrame
        DataFrame containing time series data with 'ds' (date) and 'y' (value) columns.
    days_obsoletes : int
        Number of days to look back from the last date to check for obsolescence.

    Returns
    -------
    bool
        True if all values in the recent period (last 'days_obsoletes' days) are zero,
        False otherwise.
    """
    last_date = group["ds"].max()
    cutoff_date = last_date - pd.Timedelta(days=days_obsoletes)
    recent_data = group[group["ds"] >= cutoff_date]
    return (recent_data["y"] == 0).all()


def assess_comparability(
    df: pd.DataFrame,
    features: Union[str, List[str]],
    group_col: str = "group",
    treatment: Union[str, List[str]] = "treatment",
    control: str = "control",
):
    """

    Assesses the statistical balance (comparability) of treatment and control groups
    by calculating Cohen's d effect size for comparing groups across specified features.

    This function computes Cohen's d, a standardized measure of effect size, to quantify
    the difference between treatment and control groups for each specified feature.
    High values of Cohen's d (typically > 0.4 or 0.5) suggest **group imbalance** (selection bias)
    on that feature. Cohen's d represents the difference between group means in terms of
    pooled standard deviation units.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to analyze.
    features : str or list of str
        Feature column name(s) to analyze. If a string is provided, it will be
        converted to a single-element list.
    group_col : str, optional (default="group")
        Name of the column that identifies the groups.
    treatment : str or list of str, optional (default="treatment")
        Name(s) of the treatment group(s) to compare against control.
        If a string is provided, it will be converted to a single-element list.
    control : str, optional (default="control")
        Name of the control group to use as baseline for comparison.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['group', 'feature', 'cohen_d'] containing
        Cohen's d values for each treatment group and feature combination.

    Raises
    ------
    ValueError
        If any specified features are not found in the DataFrame columns.

    Notes
    -----
    Cohen's d is calculated as:
    d = |mean_treatment - mean_control| / sqrt(pooled_variance)

    Where pooled_variance = (var_treatment + var_control) / 2

    Interpretation guidelines for Cohen's d:
    - 0.2: small effect
    - 0.5: medium effect
    - 0.8: large effect
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    if isinstance(features, str):
        features = [features]
    if isinstance(treatment, str):
        treatment = [treatment]

    if not all(feature in df.columns for feature in features):
        missing_features = [
            feature for feature in features if feature not in df.columns
        ]
        raise ValueError(f"Features not found in DataFrame columns: {missing_features}")
    if group_col not in df.columns:
        raise ValueError(f"Group column not found in DataFrame: {group_col!r}")

    available_groups = set(df[group_col].dropna().unique())
    missing_groups = [
        group for group in [control, *treatment] if group not in available_groups
    ]
    if missing_groups:
        raise ValueError(f"Groups not found in DataFrame: {missing_groups}")

    results = []

    stats = df.groupby(group_col, group_keys=True)[features].agg(["mean", "var"])

    for group in treatment:

        for feature in features:

            mean_diff = np.abs(
                stats.loc[control][feature]["mean"] - stats.loc[group][feature]["mean"]
            )
            pooled_var = (
                stats.loc[control][feature]["var"] + stats.loc[group][feature]["var"]
            ) / 2
            cohen_d_value = mean_diff / np.sqrt(pooled_var)

            results.append(
                {"group": group, "feature": feature, "cohen_d": cohen_d_value}
            )

    return pd.DataFrame(results)
