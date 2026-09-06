# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import math
from collections import Counter
from typing import List, Union

import numpy as np
import scipy.signal as signal

from .diagnostic import trend_significance


def sample_entropy(
    X: Union[np.ndarray, List[float]],
    m: int = 1,
    tolerance: float = None,
    detrend: bool = False,
) -> np.ndarray:
    """
    Compute the Sample Entropy (SampEn) of a 1D time series.

    Sample Entropy is a measure of complexity or irregularity in a time series.
    It quantifies the likelihood that similar patterns in the data will not be followed by additional similar patterns.

    Parameters
    ----------
    X : Union[np.ndarray, List[float]]
        1D time series data.
    m : int, optional, default=1
        Length of sequences to be compared (embedding dimension).
    tolerance : float, optional, default=None
        Tolerance for accepting matches. If None, it is set to 0.2 * std(X).
    detrend : bool, optional, default=False
        Whether to detrend the series before calculating entropy.

    Returns
    -------
    float
        The Sample Entropy of the time series. Returns np.nan if A or B is zero.

    Notes
    -----
    - SampEn is less biased than Approximate Entropy (ApEn) and does not count self-matches.
    - Higher SampEn values indicate more complexity and irregularity in the time series.
    - Employs Chebyshev distance (maximum norm) for pattern comparison
    - The function assumes the input time series is 1-dimensional.
    - If either A or B is zero, SampEn is undefined and np.nan is returned.

    References
    ----------
    - Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using
      approximate entropy and sample entropy. American Journal of Physiology-Heart and
      Circulatory Physiology, 278(6), H2039-H2049.
    - Lake, D. E., Richman, J. S., Griffin, M. P., & Moorman, J. R. (2002). Sample entropy
      analysis of neonatal heart rate variability. American Journal of Physiology-Regulatory,
      Integrative and Comparative Physiology, 283(3), R789-R797.
    """

    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if not np.isfinite(X).all():
        raise ValueError("Input data must contain only finite values")

    if detrend:
        r_squared, p_value = trend_significance(X)
        if r_squared > 0.3 and p_value < 0.05:
            X = signal.detrend(X, type="linear")
        else:
            X = signal.detrend(X, type="constant")

    n = X.shape[0]

    if tolerance is None:
        tolerance = 0.2 * np.std(X)

    if m < 1:
        raise ValueError("m must be a positive integer")

    if tolerance <= 0:
        raise ValueError("tolerance must be a positive float")

    if m >= n:
        raise ValueError("m must be smaller than the length of the time series")

    Xm = np.array([X[i : i + m] for i in range(n - m + 1)])

    Xm1 = np.array([X[i : i + m + 1] for i in range(n - m)])

    def count_matches(X_templates, tol):
        """
        Count the number of matching template pairs within the given tolerance. Chebyshev distance is used.

        Parameters
        ----------
        X_templates : ndarray, shape (N, m) or (N, m+1)
            Array of template vectors.
        tol : float
            Tolerance for accepting matches.
        Returns
        -------
        count : int
            Number of matching template pairs.
        """

        count = 0
        N = len(X_templates)
        for i in range(N):
            diff = np.abs(X_templates[i] - X_templates[i + 1 :])
            max_diff = np.max(diff, axis=1)
            count += np.sum(max_diff < tol)
        return count

    B = count_matches(Xm, tolerance)

    A = count_matches(Xm1, tolerance)

    if A > 0 and B > 0:
        sampen = -np.log(A / B)
    else:
        sampen = np.nan

    return sampen


def regularity_index(
    X: Union[np.ndarray, List[float]],
    m: int = 1,
    tolerance=None,
    detrend: bool = False,
) -> float:
    """
    Calculate the Regularity Index based on Sample Entropy (SampEn).

    This function measures the temporal regularity and predictability of a time series by
    inverting the Sample Entropy. It quantifies how consistent the values and patterns
    are over time, considering both magnitude and sequential relationships.

    The regularity is computed as: 1 / exp(SampEn), where higher values indicate
    more regular and predictable behavior.

    Parameters
    ----------
    X : Union[np.ndarray, List[float]]
        The time series data (e.g., prices, returns, measurements).
    m : int, optional, default=1
        The embedding dimension (length of sequences to compare).
    tolerance : float, optional, default=None
        The similarity criterion for matching patterns. If None, defaults to 0.2 * std(X).
    detrend : bool, optional, default=False
        Whether to detrend the series before calculating entropy.

    Returns
    -------
    float
        The Regularity Index, where:
        - Values close to 1: High regularity/predictability (consistent patterns)
        - Values close to 0: Low regularity/predictability (irregular/complex behavior)

    Notes
    -----
    - Uses Sample Entropy which considers actual value magnitudes and distances
    - Higher tolerance allows more variation in "similar" patterns
    - Complementary to ordinal-based measures like theoretical_limit()
    """
    hrate = sample_entropy(X, m=m, tolerance=tolerance, detrend=detrend)
    return 1 / np.exp(hrate)


def permutation_entropy(
    X: Union[np.ndarray, List[float]],
    m: int = 3,
    delay: int = 1,
    normalize=True,
):
    """
    Calculate the Permutation Entropy of a time series.

    Permutation Entropy (PE) is a complexity measure that quantifies the regularity
    and predictability of a time series by analyzing ordinal patterns. It focuses on
    the relative order of values rather than their actual magnitudes, making it robust
    to noise and outliers.

    Parameters
    ----------
    X : Union[np.ndarray, List[float]]
        Time series data (e.g., closing prices, measurements).
    m : int, optional, default=3
        The embedding dimension (length of ordinal patterns to analyze).
        Common values are 3-7, with 3-5 being most typical.
    delay : int, optional, default=1
        The time delay (spacing between elements in patterns).
        delay=1 uses consecutive elements.
    normalize : bool, optional, default=True
        If True, normalize PE by log₂(m!) to get values in [0,1].
        If False, return raw entropy values.

    Returns
    -------
    float
        The Permutation Entropy of the time series:
        - If normalized: 0 (completely regular) to 1 (completely random)
        - If not normalized: 0 to log₂(m!)

    Notes
    -----
    - PE analyzes ordinal patterns by comparing relative ordering of m consecutive values
    - Higher PE values indicate more complexity/randomness in ordinal structure
    - Lower PE values suggest more regular/predictable ordinal patterns
    - Robust to noise and non-linear dynamics
    - Time complexity: O(N×m×log(m)) where N is series length
    - Requires at least (m-1)×delay + 1 data points

    References
    ----------
    - Bandt, C., & Pompe, B. (2002). Permutation entropy: A natural complexity
      measure for time series. Physical Review Letters, 88(17), 174102.
    - Zanin, M., Zunino, L., Rosso, O. A., & Papo, D. (2012). Permutation entropy
      and its main biomedical and econophysics applications: a comprehensive review.
      Entropy, 14(8), 1553-1577.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if m < 2:
        raise ValueError("m must be at least 2")
    if delay < 1:
        raise ValueError("delay must be at least 1")
    if len(X) < (m - 1) * delay + 1:
        raise ValueError("Time series is too short for the given m and delay")

    N = X.shape[0] - delay * (m - 1)
    window_indices = [np.arange(i, i + delay * m, delay) for i in range(N)]
    X = np.argsort(X[window_indices], axis=1)
    patterns = Counter(map(tuple, X))
    probs = {k: v / sum(patterns.values()) for k, v in patterns.items()}
    probs = np.array(list(probs.values()))
    pe = -np.sum(probs * np.log2(probs))
    return pe / np.log2(math.factorial(m)) if normalize else pe


def theoretical_limit(
    X: Union[np.ndarray, List[float]],
    m: int = 3,
    delay: int = 1,
) -> float:
    """
    Calculates the theoretical upper limit of predictability (Πmax) for a time series based on ordinal patterns.

    This function computes the maximum achievable predictability by analyzing the structural
    complexity of ordinal patterns in the time series, independent of magnitude. It uses
    normalized Permutation Entropy: Πmax = 1 - PE_norm.

    The theoretical limit represents the upper bound of predictability that any forecasting
    method could achieve if it perfectly captured all ordinal patterns in the data, ignoring
    actual value magnitudes.

    Parameters
    ----------
    X : Union[np.ndarray, List[float]]
        The time series data.
    m : int, optional, default=3
        The embedding dimension (length of ordinal patterns to analyze).
    delay : int, optional, default=1
        The delay (spacing between elements in patterns).

    Returns
    -------
    float
        The theoretical predictability limit (Πmax) for the time series, ranging from 0 to 1:
        - 0: Completely random ordinal patterns (maximum complexity)
        - 1: Perfectly regular ordinal patterns (minimum complexity)

    Notes
    -----
    - This is a **theoretical upper bound** based solely on ordinal structure of the series
    - The measure ignores magnitudes, focusing only on directional patterns
    - Higher values indicate more regular/predictable ordinal behavior
    - Serves as a benchmark for comparing actual forecasting performance
    - Based on Permutation Entropy theory and information-theoretic limits

    References
    ----------
    - Bandt, C., & Pompe, B. (2002). Permutation entropy: A natural complexity
      measure for time series. Physical Review Letters, 88(17), 174102.
    - Song, C., Qu, Z., Blumm, N., & Barabási, A. L. (2010). Limits of
        predictability in human mobility. Science, 327(5968), 1018-1021.
    """
    pe = permutation_entropy(X, m=m, delay=delay, normalize=True)

    return 1 - pe

