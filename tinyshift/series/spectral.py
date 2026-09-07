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

    if detrend not in ("linear", "constant", "none", None):
        raise ValueError("'detrend' must be one of {'linear', 'constant', 'none'}.")

    if detrend in ("linear", "constant"):
        scale = max(1.0, float(np.max(np.abs(signal))))
        tolerance = np.finfo(np.float64).eps * scale * len(signal)
        if np.ptp(signal) <= tolerance:
            return np.zeros_like(signal)

    if detrend == "linear":
        x = np.arange(len(signal), dtype=np.float64)
        coefficients = np.polyfit(x, signal, 1)
        signal = signal - np.polyval(coefficients, x)

    elif detrend == "constant":
        signal = signal - np.mean(signal)

    elif detrend in ("none", None):
        pass

    return signal


def _prepare_spectrum(
    series: ArrayLike,
    detrend: str = "linear",
    method: str = "periodogram",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Prepare a cleaned signal and calculate its power spectrum."""
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


def _spectral_power_distribution(
    X: ArrayLike,
    detrend: str = "linear",
) -> Tuple[np.ndarray, int]:
    """Return the normalized non-DC spectral power from _prepare_spectrum."""
    _, power, _ = _prepare_spectrum(
        X,
        detrend=detrend,
        method="periodogram",
    )

    # Remove zero-frequency / DC component.
    power = power[1:]

    total_power = np.sum(power)

    if total_power <= np.finfo(float).eps:
        return np.array([], dtype=np.float64), 0

    power_distribution = power / total_power

    n_frequencies = len(power_distribution)

    return power_distribution, n_frequencies


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
    power_distribution, n_frequencies = _spectral_power_distribution(
        X,
        detrend=detrend,
    )

    if n_frequencies <= 1:
        return np.nan

    # Ignore zero-probability bins because log2(0) is undefined.
    positive_probabilities = power_distribution[power_distribution > 0]

    # Shannon entropy of the spectral power distribution.
    spectral_entropy = -np.sum(positive_probabilities * np.log2(positive_probabilities))

    max_spectral_entropy = np.log2(n_frequencies)

    normalized_entropy = spectral_entropy / max_spectral_entropy

    # ForeCA reverses entropy:
    # 0 = diffuse / uncertain spectrum
    # 1 = concentrated / structured spectrum
    forecastability = 1.0 - normalized_entropy

    return float(np.clip(forecastability, 0.0, 1.0))


def spectral_concentration(
    X: ArrayLike,
    detrend: str = "linear",
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

    Returns
    -------
    float
        Normalized spectral concentration in the range [0, 1].

        0 means power is approximately uniformly distributed, while 1 means
        power is concentrated in one spectral component.
    """
    power_distribution, n_frequencies = _spectral_power_distribution(
        X,
        detrend=detrend,
    )

    if n_frequencies <= 1:
        return np.nan

    # Herfindahl-Hirschman / Simpson concentration index.
    concentration = np.sum(power_distribution**2)

    # Theoretical bounds:
    # - minimum: power uniformly distributed across all frequencies.
    # - maximum: all power concentrated in a single frequency.
    min_concentration = 1.0 / n_frequencies
    max_concentration = 1.0

    # Min-max normalization
    normalized_concentration = (concentration - min_concentration) / (
        max_concentration - min_concentration
    )

    return float(np.clip(normalized_concentration, 0.0, 1.0))
