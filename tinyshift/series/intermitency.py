from typing import List, Tuple, Union

import numpy as np
import pandas as pd

ArrayLike = Union[
    np.ndarray,
    List[float],
    pd.Series,
]


def _prepare_demand(
    X: ArrayLike,
) -> np.ndarray:
    """
    Validate and prepare a univariate demand series.

    Parameters
    ----------
    X : numpy.ndarray, list of float, or pandas.Series
        Input demand series.

    Returns
    -------
    numpy.ndarray
        One-dimensional array containing finite, non-negative demand values.

    Raises
    ------
    ValueError
        If the input is not one-dimensional, contains fewer than two
        observations, contains non-finite values, or contains negative values.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional.")

    if X.size < 2:
        raise ValueError("Input data must contain at least two observations.")

    if not np.isfinite(X).all():
        raise ValueError("Input data must contain only finite values.")

    if np.any(X < 0):
        raise ValueError("Demand values must be non-negative.")

    return X


def inter_demand_intervals(
    X: ArrayLike,
) -> np.ndarray:
    """
    Calculate the intervals between consecutive non-zero demand occurrences.

    The inter-demand interval measures the number of observation steps between
    consecutive periods with positive demand. It describes the temporal spacing
    of demand occurrences and can be used to assess demand intermittency.

    Parameters
    ----------
    X : numpy.ndarray, list of float, or pandas.Series
        One-dimensional non-negative demand series.

    Returns
    -------
    numpy.ndarray
        Integer array containing the distances, in observations, between
        consecutive positive-demand occurrences.

        If fewer than two positive-demand observations are present, an empty
        array is returned.

    Examples
    --------
    >>> X = [0, 2, 0, 0, 5, 0, 3]
    >>> inter_demand_intervals(X)
    array([3, 2])

    Notes
    -----
    A demand occurrence is defined as an observation strictly greater than
    zero.

    Returned intervals are expressed in numbers of observations. Therefore,
    the interpretation depends on the sampling frequency of the input series.
    For daily data, an interval of 7 corresponds to seven days; for hourly
    data, it corresponds to seven hours.
    """
    X = _prepare_demand(X)

    demand_indices = np.flatnonzero(X > 0)

    if demand_indices.size < 2:
        return np.array([], dtype=int)

    return np.diff(demand_indices)


def zero_proportion(X) -> float:
    """
    Calculate the proportion of zero-demand observations.

    Parameters
    ----------
    X : array-like
        Non-negative demand series.

    Returns
    -------
    float
        Fraction of observations with zero demand, ranging from 0 to 1.
    """
    X = _prepare_demand(X)
    return float(np.mean(X == 0))


def inter_demand_interval_stats(X) -> tuple[float, float]:
    """
    Calculate summary statistics of inter-demand intervals.

    Returns
    -------
    tuple of float
        Mean and standard deviation of the distances between consecutive
        positive-demand occurrences.
    """
    intervals = inter_demand_intervals(X)

    if intervals.size == 0:
        return float("nan"), float("nan")

    return (
        float(np.mean(intervals)),
        float(np.std(intervals, ddof=0)),
    )


def inter_demand_interval_cv(X) -> float:
    """
    Calculate the coefficient of variation of inter-demand intervals.

    The metric measures how irregularly positive-demand occurrences are
    distributed over time.

    Values near zero indicate relatively regular spacing, while larger
    values indicate more irregular occurrence patterns.
    """
    intervals = inter_demand_intervals(X)

    if intervals.size < 2:
        return float("nan")

    mean_interval = np.mean(intervals)

    if mean_interval == 0:
        return float("nan")

    return float(np.std(intervals, ddof=0) / mean_interval)


def average_demand_interval(
    X: ArrayLike,
) -> float:
    """
    Calculate the Average Demand Interval (ADI).

    ADI measures the average number of observation periods between non-zero
    demand occurrences. Larger values indicate more intermittent demand,
    while values closer to one indicate that demand occurs more frequently.

    It is defined as

    .. math::

        ADI = \\frac{N}{N_{+}},

    where :math:`N` is the total number of observations and :math:`N_{+}` is
    the number of observations with positive demand.

    Parameters
    ----------
    X : numpy.ndarray, list of float, or pandas.Series
        One-dimensional non-negative demand series.

    Returns
    -------
    float
        Average Demand Interval.

        A value of ``1.0`` indicates positive demand at every observation.
        Larger values indicate increasingly sparse demand occurrences.

        ``np.inf`` is returned when the series contains no positive demand.

    Examples
    --------
    >>> X = [0, 0, 4, 0, 0, 0, 3, 0]
    >>> average_demand_interval(X)
    4.0

    Notes
    -----
    ADI describes the frequency of demand occurrence rather than the magnitude
    of demand when it occurs.

    It is commonly combined with the squared coefficient of variation of
    positive demand sizes to characterize intermittent demand patterns.
    """
    X = _prepare_demand(X)

    n_demand = np.count_nonzero(X > 0)

    if n_demand == 0:
        return float("inf")

    return float(X.size / n_demand)


def squared_coefficient_of_variation(
    X: ArrayLike,
) -> float:
    """
    Calculate the squared coefficient of variation of positive demand.

    The squared coefficient of variation (CV²) measures the relative
    variability of demand magnitude conditional on demand occurring.

    It is defined as

    .. math::

        CV^2 =
        \\left(
            \\frac{\\sigma_{+}}{\\mu_{+}}
        \\right)^2,

    where :math:`\\mu_{+}` and :math:`\\sigma_{+}` are the mean and standard
    deviation of the strictly positive demand observations.

    Parameters
    ----------
    X : numpy.ndarray, list of float, or pandas.Series
        One-dimensional non-negative demand series.

    Returns
    -------
    float
        Squared coefficient of variation of positive demand values.

        ``np.nan`` is returned when no positive demand is present.

    Examples
    --------
    >>> X = [0, 2, 0, 4, 0, 6]
    >>> squared_coefficient_of_variation(X)
    0.16666666666666666

    Notes
    -----
    Zero-demand observations are excluded because CV² is intended to describe
    variation in demand size conditional on demand occurring.

    A value near zero indicates that positive demand sizes are relatively
    stable. Larger values indicate greater variability in demand magnitude.

    The population standard deviation (``ddof=0``) is used.
    """
    X = _prepare_demand(X)

    positive_demand = X[X > 0]

    if positive_demand.size == 0:
        return float("nan")

    mean_demand = np.mean(positive_demand)

    if mean_demand == 0:
        return float("nan")

    cv = (
        np.std(
            positive_demand,
            ddof=0,
        )
        / mean_demand
    )

    return float(cv**2)


def adi_cv(
    X: ArrayLike,
) -> Tuple[float, float]:
    """
    Calculate ADI and CV² for an intermittent demand series.

    The Average Demand Interval (ADI) measures how frequently positive demand
    occurs, while the squared coefficient of variation (CV²) measures the
    variability of demand magnitude conditional on demand occurring.

    Together, these statistics provide complementary information about the
    occurrence and size dimensions of intermittent demand.

    Parameters
    ----------
    X : numpy.ndarray, list of float, or pandas.Series
        One-dimensional non-negative demand series.

    Returns
    -------
    tuple of float
        A tuple ``(adi, cv2)`` containing:

        - ``adi``: Average Demand Interval.
        - ``cv2``: squared coefficient of variation of positive demand.

    Examples
    --------
    >>> X = [0, 2, 0, 0, 4, 0, 6, 0]
    >>> adi, cv2 = adi_cv(X)
    >>> adi
    2.6666666666666665

    Notes
    -----
    ADI captures demand occurrence sparsity, whereas CV² captures variability
    in positive demand sizes.

    These two dimensions are commonly used together to characterize demand
    patterns such as smooth, intermittent, erratic, and lumpy demand.
    """
    X = _prepare_demand(X)

    return (
        average_demand_interval(X),
        squared_coefficient_of_variation(X),
    )


def classify_intermittency(
    X: ArrayLike,
    adi_threshold: float = 1.32,
    cv2_threshold: float = 0.49,
) -> str:
    """
    Classify a demand series according to its intermittency pattern.

    The classification combines the Average Demand Interval (ADI) and the
    squared coefficient of variation (CV²) to distinguish between four demand
    patterns:

    ``"smooth"``
        Demand occurs frequently and positive demand sizes have relatively
        low variability.

    ``"intermittent"``
        Demand occurs infrequently, but positive demand sizes have relatively
        low variability.

    ``"erratic"``
        Demand occurs frequently, but positive demand sizes have relatively
        high variability.

    ``"lumpy"``
        Demand occurs infrequently and positive demand sizes have relatively
        high variability.

    Parameters
    ----------
    X : numpy.ndarray, list of float, or pandas.Series
        One-dimensional non-negative demand series.

    adi_threshold : float, default=1.32
        Threshold separating frequent from intermittent demand occurrence.

    cv2_threshold : float, default=0.49
        Threshold separating low from high variability in positive demand
        magnitude.

    Returns
    -------
    str
        One of ``"smooth"``, ``"intermittent"``, ``"erratic"``, or
        ``"lumpy"``.

    Raises
    ------
    ValueError
        If either threshold is not strictly positive.

        If the series contains no positive demand observations, because CV²
        and the resulting demand classification are undefined.

    Examples
    --------
    >>> X = [0, 0, 5, 0, 0, 6, 0, 0, 5]
    >>> classify_intermittency(X)
    'intermittent'

    Notes
    -----
    The default thresholds ``ADI = 1.32`` and ``CV² = 0.49`` correspond to
    the commonly used ADI-CV² demand categorization proposed for intermittent
    demand forecasting.

    The thresholds should not be interpreted as universal statistical
    boundaries. They represent a practical classification rule and may be
    adjusted for specific applications.

    Boundary values are assigned to the higher category. Therefore,
    ``ADI >= adi_threshold`` is considered intermittent occurrence and
    ``CV² >= cv2_threshold`` is considered high variability.
    """
    if adi_threshold <= 0:
        raise ValueError("'adi_threshold' must be positive, " f"got {adi_threshold}.")

    if cv2_threshold <= 0:
        raise ValueError("'cv2_threshold' must be positive, " f"got {cv2_threshold}.")

    X = _prepare_demand(X)

    adi, cv2 = adi_cv(X)

    if np.isnan(cv2):
        raise ValueError(
            "Intermittency classification requires at least "
            "one positive demand observation."
        )

    if adi < adi_threshold:
        if cv2 < cv2_threshold:
            return "smooth"

        return "erratic"

    if cv2 < cv2_threshold:
        return "intermittent"

    return "lumpy"
