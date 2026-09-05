# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .spectral import _prepare_spectrum

SeriesLike = Union[
    np.ndarray,
    List[float],
    pd.Series,
]

SeasonalInput = Union[
    SeriesLike,
    pd.DataFrame,
]


class SeasonalPeriodDetector:
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

    unique_id_col : str, default="unique_id"
        Column identifying individual series when fitting panel data.

    target_col : str, optional
        Target column used when fitting a DataFrame.

        If omitted, ``"y"`` is preferred when present. Otherwise, the target
        is inferred when exactly one numeric column exists besides
        ``unique_id_col``.

    Attributes
    ----------
    periods_ : list of int or dict
        Detected candidate periods after calling :meth:`fit`.

        For a single series, this is a list of periods.

        For panel data, this is a dictionary mapping each unique ID to its
        detected periods.

    frequencies_ : numpy.ndarray or dict
        Fourier frequencies computed during fitting.

        For panel data, values are stored per unique ID.

    power_ : numpy.ndarray or dict
        Spectral power associated with ``frequencies_``.

    peaks_ : numpy.ndarray or dict
        Indices of the significant spectral peaks retained before conversion
        into candidate periods.

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
    Detect seasonal periods in a single daily series:

    >>> detector = SeasonalPeriodDetector(top_k=2)
    >>> detector.fit(y)
    SeasonalPeriodDetector(...)
    >>> detector.periods_
    [7, 30]

    Inspect the fitted spectrum:

    >>> detector.frequencies_
    array([...])

    >>> detector.power_
    array([...])

    Reuse the same configuration across multiple series:

    >>> detector = SeasonalPeriodDetector(
    ...     top_k=3,
    ...     detrend="linear",
    ... )
    >>> periods_a = detector.fit(series_a).periods_
    >>> periods_b = detector.fit(series_b).periods_

    Detect periods for panel data:

    >>> detector = SeasonalPeriodDetector(
    ...     unique_id_col="unique_id",
    ...     target_col="y",
    ... )
    >>> detector.fit(data)
    SeasonalPeriodDetector(...)
    >>> detector.periods_
    {
        "series_a": [7],
        "series_b": [7, 30],
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
        unique_id_col: str = "unique_id",
        target_col: Optional[str] = None,
    ) -> None:
        self.top_k = top_k
        self.noise_threshold_factor = noise_threshold_factor
        self.fallback = fallback
        self.detrend = detrend
        self.unique_id_col = unique_id_col
        self.target_col = target_col

        self._validate_params()

    def __repr__(self) -> str:
        return (
            "SeasonalPeriodDetector("
            f"top_k={self.top_k}, "
            f"noise_threshold_factor={self.noise_threshold_factor}, "
            f"fallback={self.fallback!r}, "
            f"detrend={self.detrend!r}"
            ")"
        )

    def _validate_params(self) -> None:
        """Validate detector configuration."""
        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError(
                "'top_k' must be a positive integer, " f"got {self.top_k!r}."
            )

        if self.noise_threshold_factor <= 0:
            raise ValueError(
                "'noise_threshold_factor' must be positive, "
                f"got {self.noise_threshold_factor!r}."
            )

        if self.detrend not in {"linear", "constant", "none"}:
            raise ValueError(
                "'detrend' must be one of " "{'linear', 'constant', 'none'}."
            )

    def _normalize_fallback(self) -> List[int]:
        """Return the configured fallback as a list."""
        if self.fallback is None:
            return []

        if isinstance(self.fallback, int):
            return [self.fallback]

        return list(self.fallback)

    def _resolve_target_column(
        self,
        data: pd.DataFrame,
    ) -> str:
        """
        Resolve the target column for panel time-series input.
        """
        if self.unique_id_col not in data.columns:
            raise ValueError(
                "DataFrame must contain the unique ID column "
                f"{self.unique_id_col!r}."
            )

        if self.target_col is not None:
            if self.target_col not in data.columns:
                raise ValueError(
                    "DataFrame does not contain target column " f"{self.target_col!r}."
                )

            return self.target_col

        if "y" in data.columns:
            return "y"

        numeric_columns = [
            column
            for column in data.columns
            if (
                column != self.unique_id_col
                and pd.api.types.is_numeric_dtype(data[column])
            )
        ]

        if len(numeric_columns) != 1:
            raise ValueError(
                "Could not infer the target column. "
                "Provide `target_col`, include a column named 'y', "
                "or include exactly one numeric column besides "
                f"{self.unique_id_col!r}."
            )

        return numeric_columns[0]

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
        frequencies, power, n_observations = _prepare_spectrum(
            series,
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

        return {
            "periods": periods,
            "frequencies": frequencies,
            "power": power,
            "peaks": peaks,
        }

    def fit(
        self,
        series: SeasonalInput,
    ) -> "SeasonalPeriodDetector":
        """
        Fit the seasonal-period detector.

        Parameters
        ----------
        series : numpy.ndarray, list of float, pandas.Series, or pandas.DataFrame
            Input time-series data.

            A one-dimensional input is treated as a single regularly sampled
            series.

            A DataFrame is treated as panel data and processed independently
            by ``unique_id_col``.

        Returns
        -------
        SeasonalPeriodDetector
            The fitted detector instance.

        Notes
        -----
        Calling ``fit`` updates the fitted attributes ``periods_``,
        ``frequencies_``, ``power_``, and ``peaks_``.
        """
        if not isinstance(series, pd.DataFrame):
            result = self._fit_single(series)

            self.periods_ = result["periods"]
            self.frequencies_ = result["frequencies"]
            self.power_ = result["power"]
            self.peaks_ = result["peaks"]

            return self

        target_col = self._resolve_target_column(series)

        results = {
            unique_id: self._fit_single(group[target_col])
            for unique_id, group in series.groupby(
                self.unique_id_col,
                sort=False,
            )
        }

        self.periods_ = {
            unique_id: result["periods"] for unique_id, result in results.items()
        }

        self.frequencies_ = {
            unique_id: result["frequencies"] for unique_id, result in results.items()
        }

        self.power_ = {
            unique_id: result["power"] for unique_id, result in results.items()
        }

        self.peaks_ = {
            unique_id: result["peaks"] for unique_id, result in results.items()
        }

        return self

    def detect(
        self,
        series: SeasonalInput,
    ) -> Union[List[int], Dict[Any, List[int]]]:
        """
        Detect candidate periods and return them directly.

        This is a convenience method equivalent to calling ``fit(series)``
        followed by accessing ``periods_``.

        Parameters
        ----------
        series : numpy.ndarray, list of float, pandas.Series, or pandas.DataFrame
            Input time-series data.

        Returns
        -------
        list of int or dict
            Detected candidate seasonal periods.
        """
        self.fit(series)
        return self.periods_
