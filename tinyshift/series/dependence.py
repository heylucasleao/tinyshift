# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from collections import Counter
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import scipy.signal as signal


def permutation_auto_mutual_information(
    X: Union[np.ndarray, List[float]],
    tau: int = 1,
    m: int = 3,
    delay: int = 1,
    normalize: bool = False,
) -> float:
    """
    Calculate the Permutation Auto-Mutual Information (PAMI) of a time series.

    PAMI measures the information dependency between a time series X and itself
    delayed by a lag 'tau', using ordinal patterns. It quantifies how much information
    the ordinal patterns at time t provide about the ordinal patterns at time t+tau.

    Parameters
    ----------
    X : Union[np.ndarray, List[float]]
        Time series data (e.g., closing prices, measurements).
    tau : int, optional, default=1
        The main time lag for calculating Auto-Mutual Information.
        Compares X(t) with X(t + tau).
    m : int, optional, default=3
        The embedding dimension (length of ordinal pattern).
    delay : int, optional, default=1
        The embedding delay (internal spacing within each pattern).
    normalize : bool, optional, default=False
        If True, normalize PAMI by the minimum entropy of the two pattern sets.

    Returns
    -------
    float
        The Permutation Auto-Mutual Information of the time series for lag 'tau'.
        Higher values indicate stronger temporal dependencies between ordinal patterns.

    Notes
    -----
    - PAMI is calculated as: I_P(X(t); X(t+τ)) = Σ P(πᵢ, πⱼ) * log₂(P(πᵢ, πⱼ) / (P(πᵢ) * P(πⱼ)))
    - Higher PAMI values indicate stronger temporal dependencies between ordinal patterns
    - Values near zero suggest independence between current and lagged patterns
    - Useful for detecting non-linear predictive relationships in time series

    References
    ----------
    - Ouyang, Gaoxiang & Li, Xiaoli. (2009). Auto Mutual Information Analysis with
        Order Patterns for Epileptic EEG. 6th International Conference on Fuzzy Systems
        and Knowledge Discovery, FSKD 2009. 5. 23-27. 10.1109/FSKD.2009.33.
    - Liang, Zhenhu & Wang, Yinghua & Ouyang, Gaoxiang & Voss, Logan & Sleigh, Jamie
        & Li, Xiaoli. (2013). Permutation auto-mutual information of electroencephalogram
        in anesthesia. Journal of Neural Engineering. 10. 026004. 10.1088/1741-2560/10/2/026004.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if m < 2:
        raise ValueError("m must be at least 2")
    if tau < 1 or delay < 1:
        raise ValueError("tau and delay must be at least 1")
    if len(X) < (m - 1) * delay + 1:
        raise ValueError("Time series is too short for the given m and delay")
    if tau + (m - 1) * delay >= len(X):
        raise ValueError("Time series is too short for the given tau, m, and delay")

    N = X.shape[0] - delay * (m - 1) - tau
    window_x = [np.arange(i, i + delay * m, delay) for i in range(N)]
    window_z = [np.arange(i + tau, i + tau + delay * m, delay) for i in range(N)]

    def generate_probabilities(patterns):
        """Helper function to generate probabilities from patterns."""
        patterns = Counter(patterns)
        total_count = sum(patterns.values())
        return {k: v / total_count for k, v in patterns.items()}

    patterns_X = np.argsort(X[window_x], axis=1)
    patterns_Z = np.argsort(X[window_z], axis=1)

    joint_patterns = list(zip(map(tuple, patterns_X), map(tuple, patterns_Z)))
    prob_joint = generate_probabilities(joint_patterns)

    patterns_X = list(map(tuple, patterns_X))
    prob_X = generate_probabilities(patterns_X)

    patterns_Z = list(map(tuple, patterns_Z))
    prob_Z = generate_probabilities(patterns_Z)

    joint_keys = list(prob_joint.keys())
    prob_joint = np.array(list(prob_joint.values()))

    shannon_X = -np.sum(list(prob_X.values()) * np.log2(list(prob_X.values())))
    shannon_Z = -np.sum(list(prob_Z.values()) * np.log2(list(prob_Z.values())))
    H_min = np.min([shannon_X, shannon_Z])

    prob_X = np.array(
        [prob_X[key[0]] for key in joint_keys]
    )  # Filtering to match joint keys
    prob_Z = np.array(
        [prob_Z[key[1]] for key in joint_keys]
    )  # Filtering to match joint keys
    pami = np.sum(
        prob_joint * np.log2(prob_joint / (prob_X * prob_Z))
    )  # Mutual Information

    return pami if not normalize else pami / H_min if H_min > 0 else 0.0


def select_pami_lag(
    values: Union[np.ndarray, List[float]],
    max_tau: int = 365,
    m: int = 3,
    delay: int = 1,
    normalize: bool = False,
    fallback: Optional[int] = None,
    return_mode: Literal["range", "point", "short_term", "value_only"] = "range",
    short_term: int = 1,
) -> Tuple[Union[int, List[int]], float, np.ndarray]:
    """
    Find the first local minimum of PAMI across candidate lags and format the optimal lag output.

    Parameters
    ----------
    values : np.ndarray or list of float
        One-dimensional time series data.
    max_tau : int, default=365
        Largest lag to evaluate.
    m : int, default=3
        Embedding dimension for permutation patterns.
    delay : int, default=1
        Embedding delay for permutation patterns.
    normalize : bool, default=False
        Whether to normalize PAMI values to the [0, 1] range.
    fallback : int, optional
        Default lag to return when no local minimum is found in the PAMI curve.
        If None and no local minimum is detected, a ValueError is raised.
    return_mode : {"range", "point", "short_term", "value_only"}, default="range"
        Determines how the optimal lag structure is formatted:
        - "range": Returns continuous window from 1 to tau (e.g., [1, 2, ..., tau]).
        - "point": Returns a list with only the selected lag (e.g., [tau]).
        - "short_term": Returns short-term consecutive lags plus the selected tau (e.g., [1, tau]).
        - "value_only": Returns the raw optimal lag integer for backward compatibility.
    short_term : int, default=1
        Number of consecutive short-term lags to include when `return_mode="short_term"`.

    Returns
    -------
    Tuple[Union[int, List[int]], float, np.ndarray]
        A tuple containing:
        - Union[int, List[int]]: Selected lag integer or list of lag indices based on `return_mode`.
        - float: The PAMI value at the selected lag.
        - np.ndarray: The array of evaluated PAMI values for lags from 1 to max_tau.

    Raises
    ------
    ValueError
        If the time series is too short for the given parameters (`max_tau < 1`),
        if no local minimum is found and `fallback` is None, or if an invalid
        `return_mode` is provided.

    Notes
    -----
    The optimal lag resolution follows this decision logic:
    1. First local minimum in the PAMI curve (`scipy.signal.find_peaks`).
    2. Explicit `fallback` value if provided and no local minimum exists.
    3. Raises `ValueError` if no local minimum exists and `fallback` is None.
    """
    values = np.asarray(values, dtype=float)
    max_valid_tau = len(values) - (m - 1) * delay - 1
    max_tau = min(max_tau, max_valid_tau)

    if max_tau < 1:
        raise ValueError("Time series is too short for the given m and delay")

    taus = np.arange(1, max_tau + 1)
    pami_values = np.array(
        [
            permutation_auto_mutual_information(
                values,
                tau=int(tau),
                m=m,
                delay=delay,
                normalize=normalize,
            )
            for tau in taus
        ]
    )

    minima, _ = signal.find_peaks(-pami_values)

    def _resolve_selected_tau() -> Tuple[int, float]:
        """Resolves optimal tau and PAMI value using local minimum or explicit fallback."""
        if len(minima) > 0:
            position = minima[0]
            return int(taus[position]), float(pami_values[position])

        if fallback is not None:
            if 1 <= fallback <= len(pami_values):
                pami_val = float(pami_values[fallback - 1])
            else:
                pami_val = np.nan
            return fallback, pami_val

        raise ValueError(
            "No local minimum was found in the PAMI curve and no explicit 'fallback' "
            "lag was provided. Provide a 'fallback' value (e.g., fallback=1) to handle "
            "flat or monotonic PAMI curves."
        )

    def _format_lags_output(tau: int) -> Union[int, List[int]]:
        """Formats selected tau into requested structure based on return_mode."""
        if return_mode == "value_only":
            return tau
        if return_mode == "point":
            return [tau]
        if return_mode == "short_term":
            short_lags = list(range(1, min(short_term, tau) + 1))
            return sorted(list(set(short_lags + [tau])))
        if return_mode == "range":
            return list(range(1, tau + 1))

        raise ValueError(
            f"Invalid return_mode '{return_mode}'. "
            "Choose from 'range', 'point', 'short_term', or 'value_only'."
        )

    selected_tau, pami_val = _resolve_selected_tau()
    lags_output = _format_lags_output(selected_tau)

    return lags_output, pami_val, pami_values

