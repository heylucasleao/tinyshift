# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from numbers import Real
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ..diagnostic import harmonic_significance
from ..spectral import _prepare_signal, _prepare_spectrum
from .base import BaseSeriesAnalyzer

SeriesLike = Union[
    np.ndarray,
    List[float],
    pd.Series,
]


class SeasonalityAnalyzer(BaseSeriesAnalyzer):
    """
    Detect dominant candidate seasonal periods from spectral peaks.

    The detector identifies recurring periodic structure in regularly sampled
    time-series data using Fourier spectral analysis.

    The input signal is detrended, transformed into the frequency domain,
    and inspected for dominant peaks in spectral power. Each significant
    frequency is converted into a candidate seasonal period using

    .. math::

        T = 1 / f,

    where :math:`f` is the Fourier frequency and :math:`T` is the period
    measured in observations.

    This class retains the intermediate spectral representation and detected peaks after fitting.
    This is useful for diagnostics, visualization, debugging, and downstream analysis.

    Parameters
    ----------
    top_k : int, default=2
        Maximum number of distinct seasonal-period candidates to retain.

        Peaks are ranked internally by spectral power. Duplicate periods
        produced by nearby frequencies are removed.

    noise_threshold_factor : float, default=2.0
        Multiplicative factor applied to the estimated spectral background
        when determining whether a peak is sufficiently strong.

        Larger values make detection more conservative, while smaller values
        permit weaker periodic components.

    fallback : int or list of int, optional
        Period or periods returned when no significant seasonal component is
        detected.

        If an integer is provided, it is converted to a single-element list.
        If ``None``, an empty list is used.

    detrend : {"linear", "constant", "none"}, default="linear"
        Detrending strategy applied before spectral analysis.

        ``"linear"``
            Remove a fitted linear trend.

        ``"constant"``
            Remove the mean.

        ``"none"``
            Analyze the original signal without detrending.

    significance_level : float, default=0.05
        P-value threshold used to retain statistically significant candidate
        periods after harmonic regression.

    Attributes
    ----------
    results_ : dict
        Mapping from each unique ID to a diagnostics dictionary containing
        candidates, significant periods, harmonic tests, and spectral details.

    Notes
    -----
    The detector identifies candidate seasonal periods; it does not prove that
    a series is seasonal.

    A strong Fourier peak can result from genuine recurring seasonality, but
    also from harmonics, transient behavior, structural breaks, finite-sample
    effects, or other repeated patterns.

    The zero-frequency component is excluded because it corresponds to the
    signal's DC level rather than a recurring cycle.

    Candidate periods longer than half the observed sample length are ignored.
    This requires approximately two observed repetitions of a candidate cycle.

    The input is assumed to be regularly sampled. Returned periods are
    therefore expressed in numbers of observations.

    Examples
    --------
    Detect periods for panel data:

    >>> analyzer = SeasonalityAnalyzer()
    >>> analyzer.fit(data, id_col="unique_id", time_col="ds", target_col="y")
    SeasonalityAnalyzer(...)
    >>> detector.results_
    {
        "series_a": {
            "candidate_periods": [7],
            "frequencies": ...,
            "power": ...,
            "peaks": ...,
        },
        "series_b": {
            "candidate_periods": [7, 30],
            "frequencies": ...,
            "power": ...,
            "peaks": ...,
        },
    }

    See Also
    --------
    foreca :
        Measures forecastability using normalized spectral entropy.

    spectral_concentration :
        Measures concentration of spectral power across frequencies.
    """

    def __init__(
        self,
        top_k: int = 2,
        noise_threshold_factor: float = 2.0,
        fallback: Optional[Union[int, List[int]]] = None,
        detrend: str = "linear",
        significance_level: float = 0.05,
    ) -> None:
        self.top_k = top_k
        self.noise_threshold_factor = noise_threshold_factor
        self.fallback = fallback
        self.detrend = detrend
        self.significance_level = significance_level

        self._validate_params()

    def __repr__(self) -> str:
        return (
            "SeasonalityAnalyzer("
            f"top_k={self.top_k}, "
            f"noise_threshold_factor={self.noise_threshold_factor}, "
            f"fallback={self.fallback!r}, "
            f"detrend={self.detrend!r}, "
            f"significance_level={self.significance_level}"
            ")"
        )

    def _validate_params(self) -> None:
        """Validate detector configuration."""
        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError(f"'top_k' must be a positive integer, got {self.top_k!r}.")

        if (
            isinstance(self.noise_threshold_factor, bool)
            or not isinstance(self.noise_threshold_factor, Real)
            or not np.isfinite(self.noise_threshold_factor)
            or self.noise_threshold_factor <= 0
        ):
            raise ValueError(
                "'noise_threshold_factor' must be positive, "
                f"got {self.noise_threshold_factor!r}."
            )

        if self.detrend not in {"linear", "constant", "none"}:
            raise ValueError("'detrend' must be one of {'linear', 'constant', 'none'}.")

        if (
            isinstance(self.significance_level, bool)
            or not isinstance(self.significance_level, Real)
            or not np.isfinite(self.significance_level)
            or not 0 < self.significance_level < 1
        ):
            raise ValueError("'significance_level' must be between 0 and 1.")

        if self.fallback is not None:
            fallback = (
                [self.fallback]
                if isinstance(self.fallback, int)
                and not isinstance(self.fallback, bool)
                else self.fallback
            )
            if not isinstance(fallback, (list, tuple)) or any(
                isinstance(period, bool) or not isinstance(period, int) or period <= 1
                for period in fallback
            ):
                raise ValueError(
                    "'fallback' must be an integer greater than 1 or a list "
                    "of integers greater than 1."
                )

    def _normalize_fallback(self) -> List[int]:
        """Return the configured fallback as a list."""
        if self.fallback is None:
            return []

        if isinstance(self.fallback, int):
            return [self.fallback]

        return list(self.fallback)

    @staticmethod
    def _spectral_background(
        power: np.ndarray,
    ) -> float:
        """
        Estimate background spectral power robustly.

        The median is preferred because large spectral peaks have less
        influence on it than on the arithmetic mean.
        """
        background = float(np.median(power))

        if background <= 0:
            background = float(np.mean(power))

        return background

    def _find_peaks(
        self,
        power: np.ndarray,
    ) -> np.ndarray:
        """
        Find significant peaks in non-DC spectral power.
        """
        if power.size == 0:
            return np.array([], dtype=int)

        background = self._spectral_background(power)

        if background <= 0:
            return np.array([], dtype=int)

        threshold = background * self.noise_threshold_factor

        peaks, _ = find_peaks(
            power,
            height=threshold,
            prominence=background,
        )

        # scipy.signal.find_peaks excludes endpoints. The last rFFT bin is
        # the Nyquist frequency for even-length signals and represents the
        # valid seasonal period 2, so evaluate that endpoint explicitly.
        if power.size == 1:
            if power[0] >= threshold:
                peaks = np.array([0], dtype=int)
        elif (
            power[-1] >= threshold
            and power[-1] > power[-2]
            and power[-1] - power[-2] >= background
        ):
            peaks = np.append(peaks, power.size - 1)

        return peaks

    @staticmethod
    def _frequency_to_period(
        frequency: float,
    ) -> Optional[int]:
        """
        Convert a positive frequency to an integer period.
        """
        if frequency <= 0:
            return None

        period = int(round(1.0 / frequency))

        if period <= 1:
            return None

        return period

    def _extract_periods(
        self,
        frequencies: np.ndarray,
        power: np.ndarray,
        peaks: np.ndarray,
        n_observations: int,
    ) -> List[int]:
        """
        Convert ranked spectral peaks into unique candidate periods.
        """
        ranked_peaks = sorted(
            peaks,
            key=lambda index: power[index],
            reverse=True,
        )

        periods: List[int] = []

        for peak in ranked_peaks:
            period = self._frequency_to_period(frequencies[peak])

            if period is None:
                continue

            if period > n_observations // 2:
                continue

            if period in periods:
                continue

            periods.append(period)

            if len(periods) >= self.top_k:
                break

        return sorted(periods)

    def _fit_single(
        self,
        series: SeriesLike,
    ) -> Dict[str, Any]:
        """
        Fit the detector to a single time series.
        """
        values = np.asarray(series, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("Input data must be 1-dimensional.")
        if np.isinf(values).any():
            raise ValueError("Input series must not contain infinite values.")
        if np.isnan(values).any():
            values = (
                pd.Series(values)
                .interpolate(method="linear", limit_direction="both")
                .to_numpy()
            )

        frequencies, power, n_observations = _prepare_spectrum(
            values,
            detrend=self.detrend,
            method="fft",
        )

        non_dc_power = power[1:]

        peaks = self._find_peaks(non_dc_power)

        if peaks.size > 0:
            # _find_peaks receives power[1:], so restore the
            # original FFT indices.
            peaks = peaks + 1

        periods = self._extract_periods(
            frequencies=frequencies,
            power=power,
            peaks=peaks,
            n_observations=n_observations,
        )

        if not periods:
            periods = self._normalize_fallback()

        detrended = _prepare_signal(values, detrend=self.detrend)
        seasonalities = {}
        for period in periods:
            f_statistic, p_value = harmonic_significance(detrended, period)
            seasonalities[period] = {
                "f_statistic": f_statistic,
                "p_value": p_value,
            }
        significant_periods = [
            period
            for period in periods
            if seasonalities[period]["p_value"] < self.significance_level
        ]

        return {
            "candidate_periods": periods,
            "significant_periods": significant_periods,
            "seasonalities": seasonalities,
            "frequencies": frequencies,
            "power": power,
            "peaks": peaks,
        }

    def summary(self) -> pd.DataFrame:
        """Return detected candidate periods with one row per series.

        Returns
        -------
        pandas.DataFrame
            ID column and detected candidate periods.

        Raises
        ------
        RuntimeError
            If the detector has not been fitted.
        """
        if not hasattr(self, "results_"):
            raise RuntimeError(
                "The detector must be fitted before calling `summary()`."
            )

        columns = ["candidate_periods", "significant_periods"]
        rows = [
            {
                self.id_col_: unique_id,
                **{column: result[column] for column in columns},
            }
            for unique_id, result in self.results_.items()
        ]
        return pd.DataFrame(rows, columns=[self.id_col_, *columns])
