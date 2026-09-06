# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import List, Union

import numpy as np
import pandas as pd

from tinyshift.utils.imports import requires_extra


def hampel_filter(
    X: Union[np.ndarray, List[float]],
    window_size: int = 3,
    factor: float = 3.0,
    scale: float = 1.4826,
) -> pd.Series:
    """
    Identify outliers using a vectorized implementation of the Hampel filter.

    The Hampel filter is a robust outlier detection method that uses the median and
    median absolute deviation (MAD) of a rolling window to identify points that
    deviate significantly from the local trend. This version uses vectorized operations
    for improved performance.

    Parameters
    ----------
    X : ndarray of shape (n_samples,) or list of float
        Input 1D data to be filtered.
    window_size : int, default=3
        Size of the trailing rolling window (must be >= 3).
    factor : float, default=3.0
        Recommended values for common distributions (95% confidence):
        - Normal distribution: 3.0 (default)
        - Laplace distribution: 2.3
        - Cauchy distribution: 3.4
        - Exponential distribution: 3.6
        - Uniform distribution: 3.9
        Number of scaled MADs from the median to consider as outlier.
    scale : float, default=1.4826
        Scaling factor for MAD to make it consistent with standard deviation.
        Recommended values for different distributions:
        - Normal distribution: 1.4826 (default)
        - Uniform distribution: 1.16
        - Laplace distribution: 2.04
        - Exponential distribution: 2.08
        - Cauchy distribution: 1.0 (MAD is already consistent)
        - These values make the MAD scale estimator consistent with the standard
        deviation for the respective distribution.

    Returns
    -------
    outliers : ndarray of shape (n_samples,)
        Boolean array indicating outliers (True) and inliers (False).

    Raises
    ------
    ValueError
        If window_size is not an integer greater than or equal to 3.
        If input data is not 1-dimensional.

    Notes
    -----
    The scale factor is chosen such that for large samples from the specified
    distribution, the median absolute deviation (MAD) multiplied by the scale
    factor approaches the standard deviation of the distribution.
    This implementation uses vectorized operations for better performance
    compared to the iterative version.
    """

    if (
        isinstance(window_size, (bool, np.bool_))
        or not isinstance(window_size, (int, np.integer))
        or window_size < 3
    ):
        raise ValueError("window_size must be an integer >= 3")
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("factor must be a positive finite number")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number")
    index = X.index if isinstance(X, pd.Series) else pd.RangeIndex(len(X))
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    n_samples = X.shape[0]
    is_outlier = np.zeros(n_samples, dtype=bool)

    start_index = window_size - 1
    center_indices = np.arange(start_index, n_samples)
    offsets = np.arange(-window_size + 1, 1)
    window_indices = center_indices[:, None] + offsets[None, :]

    if window_indices.shape[0] == 0:
        return pd.Series(is_outlier, index=index)

    windows = X[window_indices]

    medians = np.nanmedian(windows, axis=1)
    mads = np.nanmedian(np.abs(windows - medians[:, None]), axis=1)
    thresholds = factor * mads * scale
    is_outlier[center_indices] = np.abs(X[center_indices] - medians) > thresholds

    return pd.Series(is_outlier, index=index)


@requires_extra("series")
def bollinger_bands(
    X: Union[np.ndarray, List[float]],
    window_size: int = 20,
    factor: float = 2.0,
) -> pd.Series:
    """
    Feature transformer that computes the Bollinger Bands for a given time series.
    Bollinger Bands consist of a middle band (simple moving average) and two outer bands
    that are a specified number of standard deviations away from the middle band.
    The bands help identify periods of high and low volatility in the time series.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
        Time series data (e.g., closing prices).
    window_size : int, optional (default=20)
        The number of periods to use for calculating the moving average and standard deviation.
    factor : float, optional (default=2)
        The number of standard deviations to use for the upper and lower bands.

    Returns
    -------
    outliers : ndarray, shape (n_samples,)
        Boolean array indicating outliers (True) and inliers (False).

    Notes
    -----
    - The Bollinger Bands are calculated using a rolling window approach.
    - Outliers are points outside the upper or lower band.
    """
    index = X.index if isinstance(X, pd.Series) else pd.RangeIndex(len(X))
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if (
        isinstance(window_size, (bool, np.bool_))
        or not isinstance(window_size, (int, np.integer))
        or window_size < 2
    ):
        raise ValueError("window_size must be an integer >= 2")
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("factor must be a positive finite number")
    if window_size > X.shape[0]:
        raise ValueError("window_size cannot be larger than the length of X")

    from coreforecast.rolling import rolling_mean, rolling_std

    centers = rolling_mean(X, window_size=window_size)
    spreads = rolling_std(X, window_size=window_size)

    # coreforecast uses the sample standard deviation; Bollinger Bands use the
    # population standard deviation in TinyShift.
    spreads *= np.sqrt((window_size - 1) / window_size)

    first = window_size - 1
    centers[:first] = centers[first]
    spreads[:first] = spreads[first]
    bounds = np.column_stack(
        (centers - factor * spreads, centers + factor * spreads)
    )

    is_outlier = (X < bounds[:, 0]) | (X > bounds[:, 1])

    return pd.Series(is_outlier, index=index)
