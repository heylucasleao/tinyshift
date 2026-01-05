# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
from typing import List, Union


def vi(
    y_hat: Union[np.ndarray, List[float]],
    anchor: Union[np.ndarray, List[float]],
    w_s: float,
) -> np.ndarray:
    """
    Vertical Interpolation (VI) calculates the stable forecast for a specific
    target time point by linearly combining the latest original forecast (current origin)
    with the corresponding forecast value from the immediately preceding origin (the anchor).

    This function implements the core linear combination formula for vertical stability.
    The stabilization method (Partial or Full) is determined by the nature of the
    `anchor` value.

    Parameters
    ----------
    y_hat : float
        The original forecast value from the current origin.
    anchor : float
        The anchor value from the previous origin (O_i-1) corresponding to the same target time.
    w_s : float
        Weight for the corresponding forecast from the previous origin (O_i-1).
        (0 <= w_s <= 1). Higher values move the new forecast closer to the old anchor.

    Returns
    -------
    float
        The stable forecast value for the specific target time point (SF_O_i_H_j)[cite: 183].

    Notes
    --------------
    - **Partial Vertical Interpolation (PVI):** For PVI, the `anchor` parameter
      must be the **Original Forecast (F)** from the previous origin.
      This creates a pairwise comparison of forecasts.

    - **Full Vertical Interpolation (FVI):** For FVI, the `anchor` parameter
      must be the **Stabilized Forecast (SF)** from the previous origin.
      This requires a **sequential/chained pipeline** because the stabilized output of
      origin must be calculated and saved before it can be used as the anchor
      input for origin.

    References
    ----------
    - Godahewa, R., Bergmeir, C., Erkin Baz, Z., Zhu, C., Song, Z., García, S.,
      & Benavides, D. (2023). On forecast stability. International Journal of
      Forecasting, 41(4), 1539-1558.
    """

    if not (0 <= w_s <= 1):
        raise ValueError("Weight w_s must be between 0 and 1.")

    y_hat = np.asarray(y_hat)

    if y_hat.ndim != 1:
        raise ValueError("The original forecast series must be one-dimensional.")

    if anchor.ndim != 1:
        raise ValueError("The original forecast series must be one-dimensional.")

    fc = (w_s * anchor) + ((1 - w_s) * y_hat)
    return fc


def hpi(
    y_hat: Union[np.ndarray, List[float]],
    w_s: float,
) -> np.ndarray:
    """
    Horizontal Partial Interpolation (HPI) combines the original forecast of the current horizon with the
    original forecast of the previous horizon using a weight value. This technique helps stabilize
    forecasts by reducing the variability between consecutive forecast horizons.

    Parameters
    ----------
    y_hat : array-like
        List or array of original forecasts [F_H1, F_H2, F_H3, ...] representing forecasts for different horizons.
        Must be one-dimensional.
    w_s : float
        Weight for the previous horizon forecast (0 <= w_s <= 1). Higher values give more weight to
        the previous horizon, resulting in smoother forecasts.

    Returns
    -------
    numpy.ndarray
        Array of stable forecasts [SF_H1, SF_H2, SF_H3, ...] where each element is the interpolated
        forecast for the corresponding horizon.

    Raises
    ------
    ValueError
        If the input forecast series is not one-dimensional.

    Notes
    -----
    - The first forecast (SF_H1) equals the original forecast (F_H1)
    - Higher w_s values create smoother, more stable forecast trajectories
    - Lower w_s values preserve more of the original forecast dynamics

    References
    ----------
    - Godahewa, R., Bergmeir, C., Erkin Baz, Z., Zhu, C., Song, Z., García, S.,
      & Benavides, D. (2023). On forecast stability. International Journal of
      Forecasting, 41(4), 1539-1558.
    """

    if not (0 <= w_s <= 1):
        raise ValueError("Weight w_s must be between 0 and 1.")

    y_hat = np.asarray(y_hat)

    if y_hat.ndim != 1:
        raise ValueError("The original forecast series must be one-dimensional.")

    fc = y_hat.copy()

    for i in range(1, fc.shape[0]):
        fc[i] = (w_s * y_hat[i - 1]) + ((1 - w_s) * y_hat[i])
    return fc


def hfi(
    y_hat: Union[np.ndarray, List[float]],
    w_s: float,
) -> np.ndarray:
    """
    Horizontal Full Interpolation (HFI) is a forecast combination technique that blends
    the stable forecast from the previous horizon with the original forecast of the current
    horizon using a weight value. This technique helps stabilize forecasts by reducing the variability between consecutive forecast horizons.

    Parameters
    ----------
    y_hat : array-like
        List or array of original forecasts [F_H1, F_H2, F_H3, ...] from a single origin.
        Must be one-dimensional.
    w_s : float
        Weight for the previous stable forecast (0 <= w_s <= 1).
        Higher values give more weight to the previous stable forecast, creating smoother trajectories.

    -------
    numpy.ndarray
        Array of stable forecasts [SF_H1, SF_H2, SF_H3, ...] after applying HFI.
        The first element remains unchanged (SF_H1 = F_H1).

    Raises
    ------
    ValueError
        If the input forecast series is not one-dimensional.

    Notes
    -----
    - The first forecast (SF_H1) equals the original forecast (F_H1)
    - Higher w_s values create smoother, more stable forecast trajectories
    - Lower w_s values preserve more of the original forecast dynamics

    References
    ----------
    - Godahewa, R., Bergmeir, C., Erkin Baz, Z., Zhu, C., Song, Z., García, S.,
      & Benavides, D. (2023). On forecast stability. International Journal of
      Forecasting, 41(4), 1539-1558.
    """

    if not (0 <= w_s <= 1):
        raise ValueError("Weight w_s must be between 0 and 1.")

    y_hat = np.asarray(y_hat)

    if y_hat.ndim != 1:
        raise ValueError("The original forecast series must be one-dimensional.")

    fc = y_hat.copy()

    for i in range(1, fc.shape[0]):
        fc[i] = (w_s * fc[i - 1]) + ((1 - w_s) * fc[i])

    return fc
