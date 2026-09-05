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
