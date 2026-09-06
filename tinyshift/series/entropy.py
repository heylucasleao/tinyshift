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
    Calculate the Sample Entropy (SampEn) of a univariate time series.

    Sample Entropy measures the likelihood that two similar patterns of length
    ``m`` remain similar when one more observation is included. Unlike
    approximate entropy, it avoids counting self-matches, which reduces bias in
    finite samples.

    Parameters
    ----------
    X : Union[np.ndarray, List[float]]
        One-dimensional time series data.
    m : int, default=1
        Length of the sequences to compare.
    tolerance : float, optional, default=None
        Matching tolerance. If None, it is set to 0.2 times the standard
        deviation of ``X``.
    detrend : bool, default=False
        Whether to detrend the series before computing the entropy.

    Returns
    -------
    float
        The sample entropy of the series. If no valid match counts are found,
        the result is ``np.nan``.

    Notes
    -----
    - SampEn is less biased than Approximate Entropy because self-matches are
      excluded.
    - Higher values indicate greater irregularity or complexity.
    - The comparison uses the Chebyshev distance, i.e., the maximum absolute
      coordinate difference between two templates.

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
        _, r_squared, p_value = trend_significance(X)
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
    Calculate the regularity index derived from Sample Entropy.

    The regularity index is defined as the inverse exponential of the sample
    entropy, so more regular and predictable series receive larger values.

    Parameters
    ----------
    X : Union[np.ndarray, List[float]]
        One-dimensional time series data.
    m : int, optional, default=1
        Embedding dimension used by the underlying sample entropy computation.
    tolerance : float, optional, default=None
        Matching tolerance. If None, it defaults to 0.2 times the series
        standard deviation.
    detrend : bool, optional, default=False
        Whether to detrend the series before computing entropy.

    Returns
    -------
    float
        The Regularity Index, where:
        - Values close to 1: High regularity/predictability (consistent patterns)
        - Values close to 0: Low regularity/predictability (irregular/complex behavior)

    Notes
    -----
    This metric is complementary to ordinal-based measures such as
    ``theoretical_limit``: it captures the regularity of temporal patterns by
    reversing the entropy scale.
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

    Permutation Entropy measures the complexity of a signal by analyzing the
    relative ordering of values within ordinal patterns. It focuses on rank
    structure rather than absolute magnitude, which makes it robust to monotonic
    transformations and moderate noise.

    Parameters
    ----------
    X : Union[np.ndarray, List[float]]
        One-dimensional time series data.
    m : int, optional, default=3
        Embedding dimension, i.e., length of each ordinal pattern.
    delay : int, optional, default=1
        Spacing between observations inside each pattern.
    normalize : bool, optional, default=True
        If True, divide the entropy by ``log2(m!)`` so the value lies in the
        interval [0, 1].

    Returns
    -------
    float
        The permutation entropy of the series. Lower values indicate a more
        regular ordinal structure and higher predictability.

    Notes
    -----
    - The method evaluates relative ordering rather than exact magnitudes.
    - Higher values indicate more complexity and disorder in the ordinal
      structure.
    - The implementation requires at least ``(m - 1) * delay + 1`` points.

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
    Calculate the theoretical upper limit of predictability based on ordinal patterns.

    This function computes the maximum achievable predictability implied by the
    ordinal structure of the series. It is defined as one minus the normalized
    permutation entropy, so larger values indicate a more regular and predictable
    sequence.

    Parameters
    ----------
    X : Union[np.ndarray, List[float]]
        One-dimensional time series data.
    m : int, optional, default=3
        Embedding dimension used to form ordinal patterns.
    delay : int, optional, default=1
        Spacing between observations within each pattern.

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
