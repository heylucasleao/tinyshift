# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
from typing import Union


def macv(
    y_hat: Union[np.ndarray, list],
    y_hat_t_minus_1: Union[np.ndarray, list],
) -> float:
    """
    Computes the Mean Absolute Change Vertical (MAC_V) metric for forecast stability analysis.

    MAC_V quantifies the average magnitude of change between consecutive forecasts made at
    different time origins for overlapping future periods. Lower values indicate higher
    forecast stability (reduced nervousness), while higher values suggest more volatile
    forecasting behavior.

    Parameters
    y_hat : array-like
        Current forecast values (forecasts from the present origin).
    y_hat_t_minus_1 : array-like
        Previous forecast values (forecasts from the previous origin for the same future periods).

    Returns
    -------
    float
        The MAC_V value representing the mean absolute deviation between consecutive forecasts.
        Lower values indicate more stable forecasting performance.

    Raises
    ------
    ValueError
        If input arrays are not 1-dimensional, have different shapes, or contain invalid data.

    References
    ----------
    - Genov, E., Ruddick, J., Bergmeir, C., Vafaeipour, M., Coosemans, T., Garcia, S.,
      & Messagie, M. (2024). Predict. Optimize. Revise. On Forecast and Policy Stability
      in Energy Management Systems. arXiv preprint arXiv:2407.03368.
    """

    y_hat = np.asarray(y_hat)
    y_hat_t_minus_1 = np.asarray(y_hat_t_minus_1)

    if y_hat.ndim != 1 or y_hat_t_minus_1.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional (vectors).")

    if y_hat.shape != y_hat_t_minus_1.shape:
        raise ValueError(
            "Input arrays must have the same shape (length) for comparison."
        )

    absolute_change = np.abs(y_hat - y_hat_t_minus_1)
    mac_v = np.mean(absolute_change)

    return mac_v


def mach(
    y_hat: Union[np.ndarray, list],
) -> float:
    """
    Calculates the Mean Absolute Change Horizontal (MAC_H).

    This function quantifies the average magnitude of change between adjacent time steps
    within a single forecast window, measuring the 'smoothness' of the forecast curve.
    A lower value indicates higher horizontal stability.

    Parameters
    ----------
    y_hat : array-like
        The single forecast window (e.g., T+1 to T+H).

    Returns
    -------
    float
        The calculated MAC_H (absolute deviation).

    Raises
    ------
    ValueError
        If the array is not 1-dimensional or has fewer than two elements.

    References
    ----------
    - Genov, E., Ruddick, J., Bergmeir, C., Vafaeipour, M., Coosemans, T., Garcia, S.,
      & Messagie, M. (2024). Predict. Optimize. Revise. On Forecast and Policy Stability
      in Energy Management Systems. arXiv preprint arXiv:2407.03368.
    """

    y_hat = np.asarray(y_hat)

    if y_hat.ndim != 1:
        raise ValueError("Input array must be 1-dimensional (vector).")

    if y_hat.size < 2:
        raise ValueError(
            "Input array must contain at least two elements for MAC_H calculation."
        )

    absolute_change = np.abs(y_hat[1:] - y_hat[:-1])
    mac_h = np.mean(absolute_change)

    return mac_h


def mascv(
    y_train: np.ndarray, y_hat: np.ndarray, y_hat_minus_1: np.ndarray, seasonality: int
) -> float:
    """
    Calculates the Mean Absolute Scaled Change for Vertical Stability (MASC(V)).

    This function measures the scaled vertical stability by comparing forecast revisions
    to the typical seasonal variation in the training data. Lower values indicate higher
    forecast stability between consecutive forecast origins.

    Parameters
    ----------
    y_train : array-like
        Historical training data used for scaling factor calculation.
    y_hat : array-like
        Current forecast values (forecasts from the present origin).
    y_hat_minus_1 : array-like
        Previous forecast values (forecasts from the previous origin for the same future periods).
    seasonality : int
        Seasonal period for scaling (e.g., 12 for monthly data with yearly seasonality).

    Returns
    -------
    float
        The calculated MASC(V) value representing scaled vertical stability.
        Lower values indicate more stable forecasting performance.

    Raises
    ------
    ValueError
        If seasonality is <= 0, input arrays are not 1-dimensional, or y_hat and
        y_hat_minus_1 have different shapes.

    Notes
    -----
    - Returns np.inf if insufficient training data for scaling calculation.
    - Returns np.nan if forecast horizon is too short for meaningful comparison.

    References
    ----------
    - Godahewa, R., Bergmeir, C., Erkin Baz, Z., Zhu, C., Song, Z., García, S.,
      & Benavides, D. (2023). On forecast stability. International Journal of
      Forecasting, 41(4), 1539-1558.
    """
    y_train = np.asarray(y_train)
    y_hat = np.asarray(y_hat)
    y_hat_minus_1 = np.asarray(y_hat_minus_1)

    if y_hat.shape != y_hat_minus_1.shape:
        raise ValueError("y_hat and y_hat_minus_1 must have the same length.")
    if y_train.ndim != 1 or y_hat.ndim != 1 or y_hat_minus_1.ndim != 1:
        raise ValueError("All inputs must be 1D arrays (vectors).")
    if seasonality <= 0:
        raise ValueError("Seasonality must be > 0.")

    h = len(y_hat)
    n_overlap = h - 1

    if n_overlap <= 0:
        return np.nan

    def scaling_factor(y_train: np.ndarray, seasonality: int) -> tuple[float, int]:
        """
        Calculate the vertical scaling factor using absolute differences.

        Computes the sum of absolute differences between seasonal lags in the training data.
        """

        t_prime = len(y_train) - 1
        if t_prime <= seasonality:
            return 0.0, 0

        y_current = y_train[seasonality:t_prime]
        y_lagged = y_train[: t_prime - seasonality]
        scaling_sum = np.sum(np.abs(y_current - y_lagged))
        n_terms_in_sum = len(y_current)
        return scaling_sum, n_terms_in_sum

    numerator = np.sum(np.abs(y_hat[:n_overlap] - y_hat_minus_1[1:h]))
    scaling_sum, n_scale_terms = scaling_factor(y_train, seasonality)

    if n_scale_terms == 0:
        return np.inf
    denominator_factor = n_overlap / n_scale_terms
    denominator = denominator_factor * scaling_sum

    if denominator == 0:
        return 0.0 if numerator == 0 else np.inf

    return numerator / denominator


def masch(y_train: np.ndarray, y_hat: np.ndarray, seasonality: int) -> float:
    """
    Calculates the Mean Absolute Scaled Change for Horizontal Stability (MASC(H)).

    This function measures the scaled horizontal stability by comparing adjacent forecast
    differences to the typical seasonal variation in the training data. Lower values
    indicate higher horizontal forecast stability.

    Parameters
    ----------
    y_train : array-like
        Historical training data used for scaling factor calculation.
    y_hat : array-like
        Current forecast values.
    seasonality : int
        Seasonal period for scaling (e.g., 12 for monthly data with yearly seasonality).

    Returns
    -------
    float
        The calculated MASC(H) value representing scaled horizontal stability.
        Lower values indicate more stable forecasting performance.

    Raises
    ------
    ValueError
        If seasonality is <= 0 or input arrays are not 1-dimensional.

    Notes
    -----
    - Returns np.inf if insufficient training data for scaling calculation.
    - Returns np.nan if forecast horizon is too short for meaningful comparison.

    References
    ----------
    - Godahewa, R., Bergmeir, C., Erkin Baz, Z., Zhu, C., Song, Z., García, S.,
      & Benavides, D. (2023). On forecast stability. International Journal of
      Forecasting, 41(4), 1539-1558.
    """

    y_train = np.asarray(y_train)
    y_hat = np.asarray(y_hat)

    if y_train.ndim != 1 or y_hat.ndim != 1:
        raise ValueError("All inputs must be 1D arrays (vectors).")
    if seasonality <= 0:
        raise ValueError("Seasonality must be > 0.")

    h = len(y_hat)
    n_differences = h - 1

    if n_differences <= 0:
        return np.nan

    def scaling_factor(y_train: np.ndarray, seasonality: int) -> tuple[float, int]:
        """
        Calculate the horizontal scaling factor using absolute differences.

        Computes the sum of absolute differences between seasonal lags in the training data.
        """
        t = len(y_train)
        if t <= seasonality:
            return 0.0, 0

        y_current = y_train[seasonality:t]
        y_lagged = y_train[: t - seasonality]
        scaling_sum = np.sum(np.abs(y_current - y_lagged))
        n_terms_in_sum = len(y_current)
        return scaling_sum, n_terms_in_sum

    numerator = np.sum(np.abs(y_hat[1:] - y_hat[:-1]))
    scaling_sum, n_scale_terms = scaling_factor(y_train, seasonality)

    if n_scale_terms == 0:
        return np.inf

    denominator_factor = n_differences / n_scale_terms
    denominator = denominator_factor * scaling_sum

    if denominator == 0:
        return 0.0 if numerator == 0 else np.inf

    return numerator / denominator


def rmsscv(
    y_train: np.ndarray, y_hat: np.ndarray, y_hat_minus_1: np.ndarray, seasonality: int
) -> float:
    """
    Calculates the Root Mean Squared Scaled Change for Vertical Stability (RMSSC(V)).

    This function measures the scaled vertical stability using root mean squared differences
    by comparing forecast revisions to the typical seasonal variation in the training data.
    Lower values indicate higher forecast stability between consecutive forecast origins.

    Parameters
    ----------
    y_train : array-like
        Historical training data used for scaling factor calculation.
    y_hat : array-like
        Current forecast values (forecasts from the present origin).
    y_hat_minus_1 : array-like
        Previous forecast values (forecasts from the previous origin for the same future periods).
    seasonality : int
        Seasonal period for scaling (e.g., 12 for monthly data with yearly seasonality).

    Returns
    -------
    float
        The calculated RMSSC(V) value representing scaled vertical stability.
        Lower values indicate more stable forecasting performance.

    Raises
    ------
    ValueError
        If seasonality is <= 0, input arrays are not 1-dimensional, or y_hat and
        y_hat_minus_1 have different shapes.

    Notes
    -----
    - Returns np.inf if insufficient training data for scaling calculation.
    - Returns np.nan if forecast horizon is too short for meaningful comparison.

    References
    ----------
    - Godahewa, R., Bergmeir, C., Erkin Baz, Z., Zhu, C., Song, Z., García, S.,
      & Benavides, D. (2023). On forecast stability. International Journal of
      Forecasting, 41(4), 1539-1558.
    """

    y_train = np.asarray(y_train)
    y_hat = np.asarray(y_hat)
    y_hat_minus_1 = np.asarray(y_hat_minus_1)

    if y_hat.shape != y_hat_minus_1.shape:
        raise ValueError("y_hat and y_hat_minus_1 must have the same length.")
    if y_train.ndim != 1 or y_hat.ndim != 1:
        raise ValueError("All inputs must be 1D arrays (vectors).")
    if seasonality <= 0:
        raise ValueError("Seasonality must be > 0.")

    h = len(y_hat)
    n_overlap = h - 1

    if n_overlap <= 0:
        return np.nan

    def scaling_factor(y_train: np.ndarray, seasonality: int) -> tuple[float, int]:
        """
        Calculate the vertical scaling factor using squared differences.

        Computes the sum of squared differences between seasonal lags in the training data.
        """
        t_prime = len(y_train) - 1
        if t_prime <= seasonality:
            return 0.0, 0

        y_current = y_train[seasonality:t_prime]
        y_lagged = y_train[: t_prime - seasonality]
        scaling_sum = np.sum(np.square(y_current - y_lagged))
        n_terms_in_sum = len(y_current)
        return scaling_sum, n_terms_in_sum

    diff_squared_sum_num = np.sum(np.square(y_hat[:n_overlap] - y_hat_minus_1[1:h]))
    scaling_sum, n_scale_terms = scaling_factor(y_train, seasonality)

    if n_scale_terms == 0:
        return np.inf
    denominator_factor = n_overlap / n_scale_terms
    denominator = denominator_factor * scaling_sum

    if denominator == 0:
        return 0.0 if diff_squared_sum_num == 0 else np.inf

    return np.sqrt(diff_squared_sum_num / denominator)


def rmssch(y_train: np.ndarray, y_hat: np.ndarray, seasonality: int) -> float:
    """
    Calculates the Root Mean Squared Scaled Change for Horizontal Stability (RMSSC(H)).

    This function measures the scaled horizontal stability using root mean squared differences
    by comparing adjacent forecast differences to the typical seasonal variation in the training data.
    Lower values indicate higher horizontal forecast stability.

    Parameters
    ----------
    y_train : array-like
        Historical training data used for scaling factor calculation.
    y_hat : array-like
        Current forecast values.
    seasonality : int
        Seasonal period for scaling (e.g., 12 for monthly data with yearly seasonality).

    Returns
    -------
    float
        The calculated RMSSC(H) value representing scaled horizontal stability.
        Lower values indicate more stable forecasting performance.

    Raises
    ------
    ValueError
        If seasonality is <= 0 or input arrays are not 1-dimensional.

    Notes
    -----
    - Returns np.inf if insufficient training data for scaling calculation.
    - Returns np.nan if forecast horizon is too short for meaningful comparison.

    References
    ----------
    - Godahewa, R., Bergmeir, C., Erkin Baz, Z., Zhu, C., Song, Z., García, S.,
      & Benavides, D. (2023). On forecast stability. International Journal of
      Forecasting, 41(4), 1539-1558.
    """

    y_train = np.asarray(y_train)
    y_hat = np.asarray(y_hat)

    if y_train.ndim != 1 or y_hat.ndim != 1:
        raise ValueError("All inputs must be 1D arrays (vectors).")
    if seasonality <= 0:
        raise ValueError("Seasonality must be > 0.")

    h = len(y_hat)
    n_differences = h - 1

    if n_differences <= 0:
        return np.nan

    diff_squared_sum_num = np.sum(np.square(y_hat[1:] - y_hat[:-1]))

    def scaling_factor(y_train: np.ndarray, seasonality: int) -> tuple[float, int]:
        """
        Calculate the horizontal scaling factor using squared differences.

        Computes the sum of squared differences between seasonal lags in the training data.
        """
        t = len(y_train)
        if t <= seasonality:
            return 0.0, 0

        y_current = y_train[seasonality:t]
        y_lagged = y_train[: t - seasonality]
        scaling_sum = np.sum(np.square(y_current - y_lagged))
        n_terms_in_sum = len(y_current)
        return scaling_sum, n_terms_in_sum

    scaling_sum, n_scale_terms = scaling_factor(y_train, seasonality)

    if n_scale_terms == 0:
        return np.inf

    denominator_factor = n_differences / n_scale_terms
    denominator = denominator_factor * scaling_sum

    if denominator == 0:
        return 0.0 if diff_squared_sum_num == 0 else np.inf

    return np.sqrt(diff_squared_sum_num / denominator)
