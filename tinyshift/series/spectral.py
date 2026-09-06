# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.fft import rfft, rfftfreq
from scipy.signal import periodogram

ArrayLike = Union[np.ndarray, List[float], pd.Series]


def _prepare_signal(
    series: ArrayLike,
    detrend: str = "linear",
) -> np.ndarray:
    """
    Validate, clean and optionally detrend a 1D time series.
    """
    if isinstance(series, pd.Series):
        signal = series.dropna().to_numpy(dtype=np.float64)
    else:
        signal = np.asarray(series, dtype=np.float64)

        if signal.ndim != 1:
            raise ValueError("Input data must be 1-dimensional.")

        signal = signal[np.isfinite(signal)]

    if signal.ndim != 1:
        raise ValueError("Input data must be 1-dimensional.")

    if len(signal) < 4:
        raise ValueError(
            f"Input series must have at least 4 observations, got {len(signal)}."
        )

    if detrend == "linear":
        x = np.arange(len(signal), dtype=np.float64)
        coefficients = np.polyfit(x, signal, 1)
        signal = signal - np.polyval(coefficients, x)

    elif detrend == "constant":
        signal = signal - np.mean(signal)

    elif detrend in ("none", None):
        pass

    else:
        raise ValueError("'detrend' must be one of {'linear', 'constant', 'none'}.")

    return signal


def _prepare_spectrum(
    series: ArrayLike,
    detrend: str = "linear",
    method: str = "periodogram",
) -> Tuple[np.ndarray, np.ndarray, int]:
    signal = _prepare_signal(
        series,
        detrend=detrend,
    )

    n = len(signal)

    if method == "periodogram":
        frequencies, power = periodogram(
            signal,
            detrend=False,
            scaling="spectrum",
        )

    elif method == "fft":
        coefficients = rfft(signal)

        frequencies = rfftfreq(
            n,
            d=1.0,
        )

        power = np.abs(coefficients) ** 2

    else:
        raise ValueError("'method' must be one of " "{'periodogram', 'fft'}.")

    return frequencies, power, n


def foreca(
    X: ArrayLike,
    detrend: str = "linear",
) -> float:
    """
    Calculate the ForeCA omega forecastability index.

    Parameters
    ----------
    X : array-like
        Input univariate time series.
    detrend : {"linear", "constant", "none"}, default="linear"
        Detrending applied before estimating the power spectrum.

    Returns
    -------
    float
        Forecastability index between 0 and 1.

    Notes
    -----
    The measure is based on normalized Shannon spectral entropy.
    Higher values indicate a more concentrated, structured spectrum.
    """
    values = np.asarray(X, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Input series must contain only finite values.")

    _, power, _ = _prepare_spectrum(
        X,
        detrend=detrend,
        method="periodogram",
    )

    # DC should not participate after detrending.
    power = power[1:]

    total_power = np.sum(power)

    if total_power <= np.finfo(float).eps:
        return 1.0

    probabilities = power / total_power

    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(probabilities * np.log2(probabilities))

    max_entropy = np.log2(len(power))

    if max_entropy == 0:
        return np.nan

    omega = 1.0 - entropy / max_entropy

    return float(np.clip(omega, 0.0, 1.0))


def spectral_concentration(
    X: ArrayLike,
    detrend: str = "linear",
    normalize: bool = True,
) -> float:
    """
    Measure concentration of spectral power using the
    Herfindahl-Hirschman / Simpson concentration index.

    Parameters
    ----------
    X : array-like
        Input univariate time series.
    detrend : {"linear", "constant", "none"}, default="linear"
        Detrending applied before estimating the spectrum.
    normalize : bool, default=True
        If True, normalize concentration to [0, 1],
        accounting for the number of spectral bins.

    Returns
    -------
    float
        Spectral concentration.

        When normalized:
        - 0 means power is approximately uniformly distributed.
        - 1 means power is concentrated in one spectral component.
    """
    _, power, _ = _prepare_spectrum(
        X,
        detrend=detrend,
        method="periodogram",
    )

    # Remove zero-frequency / DC component.
    power = power[1:]

    total_power = np.sum(power)

    if total_power <= np.finfo(float).eps:
        return np.nan

    p = power / total_power

    concentration = np.sum(p**2)

    if not normalize:
        return float(concentration)

    k = len(p)

    if k <= 1:
        return 1.0

    minimum = 1.0 / k

    concentration = (concentration - minimum) / (1.0 - minimum)

    return float(np.clip(concentration, 0.0, 1.0))
