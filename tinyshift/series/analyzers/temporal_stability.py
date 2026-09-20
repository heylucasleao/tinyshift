# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

"""Sequential temporal-stability analysis for panel time series."""

from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .base import BaseSeriesAnalyzer


@dataclass(frozen=True)
class TemporalChange:
    """One confirmed change between consecutive temporal regimes."""

    estimated_change_time: Any
    detected_at: Any
    distance_ratio: float
    diff_mean: float
    scale_ratio: float


@dataclass(frozen=True)
class TemporalRegime:
    """Descriptive statistics for one detected temporal regime."""

    start_time: Any
    end_time: Any
    n_obs: int
    mean: float
    std: float


@dataclass(frozen=True)
class TemporalStabilityResult:
    """Changes, regimes, and sequential comparison windows for one series."""

    changes: list[TemporalChange]
    regimes: list[TemporalRegime]
    windows: list[dict[str, Any]]


@dataclass(frozen=True)
class _ChangeEvidence:
    """Internal evidence retained when a sequential change is confirmed."""

    position: int
    detected_at: Any
    distance: float
    threshold: float


class TemporalStabilityAnalyzer(BaseSeriesAnalyzer):
    """Detect persistent temporal distribution changes in panel series.

    Each fold contains ``horizon`` observations, matching the temporal geometry
    used by forecasting backtests. Its empirical distribution is compared with
    an expanding reference from the current regime using Wasserstein distance
    divided by the reference standard deviation. The threshold is calibrated
    automatically from historical pseudo-fold distances as their median plus
    three times the median absolute deviation (MAD). A numerical floor handles
    zero dispersion. Consecutive exceedances confirm a change and reset the
    reference.

    Parameters
    ----------
    horizon : int
        Number of observations in every comparison fold.
    n_windows : int, optional
        Analyze only the most recent number of eligible folds. By default all
        folds are analyzed.
    step_size : int, optional
        Number of observations between fold starts. Defaults to ``horizon``.
    min_reference_windows : int, default=4
        Minimum reference length, expressed in multiples of ``horizon``.
    confirmation_windows : int, default=2
        Consecutive threshold exceedances required to confirm a change.

    Attributes
    ----------
    results_ : dict
        Mapping from each unique ID to a :class:`TemporalStabilityResult` with
        sequential windows, confirmed changes, and detected regimes.
    id_col_, time_col_, target_col_ : str
        Column names used by the most recent call to :meth:`fit`.

    Notes
    -----
    Input rows must already be ordered by time within each series. Windows are
    measured in observations, so the analyzer assumes regularly sampled data.

    Examples
    --------
    >>> analyzer = TemporalStabilityAnalyzer(horizon=7)
    >>> summary = analyzer.fit(df).summary()
    >>> summary.head()
      unique_id start_time   end_time  reference_size  distance_ratio  status
    0         A        ...        ...              28             ...  stable
    """

    def __init__(
        self,
        horizon: int,
        n_windows: int | None = None,
        step_size: int | None = None,
        min_reference_windows: int = 4,
        confirmation_windows: int = 2,
    ) -> None:
        self.horizon = horizon
        self.n_windows = n_windows
        self.step_size = step_size
        self.min_reference_windows = min_reference_windows
        self.confirmation_windows = confirmation_windows
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate window geometry and confirmation parameters."""
        for name, value, minimum in (
            ("horizon", self.horizon, 1),
            ("min_reference_windows", self.min_reference_windows, 2),
            ("confirmation_windows", self.confirmation_windows, 1),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(
                    f"'{name}' must be an integer greater than or equal to {minimum}."
                )
        for name, value in (
            ("n_windows", self.n_windows),
            ("step_size", self.step_size),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Integral) or value < 1
            ):
                raise ValueError(f"'{name}' must be None or a positive integer.")

    @property
    def step_size_(self) -> int:
        """Return the effective distance between consecutive fold starts."""
        return self.horizon if self.step_size is None else int(self.step_size)

    @property
    def min_reference_size_(self) -> int:
        """Return the minimum reference size measured in observations."""
        return self.horizon * self.min_reference_windows

    def _validate_target(self, df: pd.DataFrame, target_col: str) -> None:
        """Require a numeric target before applying the grouped analysis."""
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise ValueError(f"Target column {target_col!r} must be numeric.")
        if not np.isfinite(df[target_col].to_numpy(dtype=float)).all():
            raise ValueError("Target values must be finite.")

    @staticmethod
    def _reference_scale(values: np.ndarray) -> float:
        """Return the stabilized reference standard deviation."""
        standard_deviation = float(np.std(values, ddof=0))
        floor = np.sqrt(np.finfo(float).eps) * max(1.0, abs(float(np.median(values))))
        return max(standard_deviation, floor)

    def _distance(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Return Wasserstein distance standardized by reference dispersion."""
        # Standardization makes shifts comparable across series: the same
        # absolute change can be material for a stable series and negligible
        # for a naturally volatile one.
        reference_scale = self._reference_scale(reference)
        return float(wasserstein_distance(reference, current) / reference_scale)

    def _resolve_threshold(self, reference: np.ndarray) -> float:
        """Calibrate a robust limit from sequential historical pseudo-folds."""
        # These pseudo-folds estimate the range of distances normally observed
        # within the active regime, without requiring a global cutoff.
        distances = []
        for stop in range(self.horizon * 2, len(reference) + 1, self.horizon):
            history = reference[: stop - self.horizon]
            current = reference[stop - self.horizon : stop]
            distances.append(self._distance(history, current))
        if not distances:
            raise ValueError(
                "The reference does not contain enough folds for calibration."
            )
        distances = np.asarray(distances)
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        numerical_floor = np.sqrt(np.finfo(float).eps) * max(1.0, median)
        return median + 3.0 * max(mad, numerical_floor)

    def _comparison_windows(
        self,
        values: np.ndarray,
        start: int,
        regime_start: int,
        pending_start: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the active reference and current comparison fold.

        Once a candidate change exists, the reference ends at its first fold.
        Freezing it there prevents candidate observations from making later
        confirmation folds look artificially similar to the reference.
        """
        reference_end = start if pending_start is None else pending_start
        reference = values[regime_start:reference_end]
        current = values[start : start + self.horizon]
        return reference, current

    def _evaluate_shift(
        self, reference: np.ndarray, current: np.ndarray
    ) -> tuple[float, float]:
        """Return a fold's standardized distance and reference-based limit."""
        distance = self._distance(reference, current)
        threshold = self._resolve_threshold(reference)
        return distance, threshold

    def _fold_starts(self, n_obs: int) -> list[int]:
        """Return eligible fold starts under the configured temporal geometry."""
        starts = list(
            range(
                self.min_reference_size_,
                n_obs - self.horizon + 1,
                self.step_size_,
            )
        )
        if self.n_windows is not None:
            starts = starts[-int(self.n_windows) :]
        return starts

    @staticmethod
    def _regime(
        values: np.ndarray,
        times: np.ndarray,
        start: int,
        end: int,
    ) -> TemporalRegime:
        """Summarize one half-open segment as a temporal regime."""
        segment = values[start:end]
        return TemporalRegime(
            start_time=times[start],
            end_time=times[end - 1],
            n_obs=len(segment),
            mean=float(np.mean(segment)),
            std=float(np.std(segment, ddof=0)),
        )

    @staticmethod
    def _prepare_series(
        values: np.ndarray, times: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize and validate one ordered series and its time index."""
        values = np.asarray(values, dtype=float)
        if values.ndim != 1:
            raise ValueError("Input data must be 1-dimensional.")
        if not np.isfinite(values).all():
            raise ValueError("Input data must contain only finite values.")
        if times is None:
            times = np.arange(len(values))
        times = np.asarray(times)
        if times.ndim != 1 or len(times) != len(values):
            raise ValueError("times must be one-dimensional and aligned with values.")
        return values, times

    def _scan_windows(
        self, values: np.ndarray, times: np.ndarray
    ) -> tuple[list[_ChangeEvidence], list[dict[str, Any]]]:
        """Evaluate folds sequentially and retain confirmed change evidence.

        Each fold is compared with the expanding reference of the active
        regime. The reference is frozen at the first threshold exceedance so
        candidate observations cannot contaminate subsequent confirmation
        checks. Once ``confirmation_windows`` consecutive folds exceed the
        robust threshold, the candidate start becomes the next regime start.

        Parameters
        ----------
        values : numpy.ndarray
            Finite one-dimensional values in temporal order.
        times : numpy.ndarray
            Time labels aligned positionally with ``values``.

        Returns
        -------
        evidence : list of _ChangeEvidence
            Confirmed change positions together with their detection time,
            standardized Wasserstein distance, and calibrated threshold.
        windows : list of dict
            One user-facing record per evaluated fold, containing its temporal
            bounds, reference size, relative distance, and detection status.
        """
        starts = self._fold_starts(len(values))
        if not starts:
            minimum = self.min_reference_size_ + self.horizon
            raise ValueError(
                f"Input data must contain at least {minimum} observations for one fold."
            )

        regime_start = 0
        pending_start: int | None = None
        confirmation_count = 0
        evidence: list[_ChangeEvidence] = []
        windows: list[dict[str, Any]] = []

        for start in starts:
            # A newly confirmed regime must accumulate enough observations
            # before it can provide a reliable reference distribution.
            if start - regime_start < self.min_reference_size_:
                continue

            reference, current = self._comparison_windows(
                values=values,
                start=start,
                regime_start=regime_start,
                pending_start=pending_start,
            )
            distance, limit = self._evaluate_shift(reference, current)
            exceeds = bool(distance > limit)

            if exceeds:
                if pending_start is None:
                    pending_start = start
                confirmation_count += 1
            else:
                pending_start = None
                confirmation_count = 0

            confirmed = bool(
                exceeds and confirmation_count >= self.confirmation_windows
            )
            status = "confirmed" if confirmed else "candidate" if exceeds else "stable"
            windows.append(
                {
                    "start_time": times[start],
                    "end_time": times[start + self.horizon - 1],
                    "reference_size": len(reference),
                    "distance_ratio": distance / limit,
                    "status": status,
                }
            )

            if confirmed:
                detected_end = start + self.horizon
                change_position = int(pending_start)
                evidence.append(
                    _ChangeEvidence(
                        position=change_position,
                        detected_at=times[detected_end - 1],
                        distance=distance,
                        threshold=limit,
                    )
                )
                regime_start = change_position
                pending_start = None
                confirmation_count = 0
        return evidence, windows

    def _build_result(
        self,
        values: np.ndarray,
        times: np.ndarray,
        evidence: list[_ChangeEvidence],
        windows: list[dict[str, Any]],
    ) -> TemporalStabilityResult:
        """Materialize regimes and changes after the sequential scan.

        Confirmed change positions partition the complete series into regimes.
        Regime statistics use every observation in each resulting segment,
        whereas ``distance_ratio`` retains the fold-level evidence available
        when the change was confirmed. Mean differences and scale ratios are
        then calculated between complete adjacent regimes.

        Parameters
        ----------
        values : numpy.ndarray
            Finite values in temporal order.
        times : numpy.ndarray
            Time labels aligned with ``values``.
        evidence : list of _ChangeEvidence
            Confirmed changes produced by :meth:`_scan_windows`.
        windows : list of dict
            Sequential fold records produced by :meth:`_scan_windows`.

        Returns
        -------
        TemporalStabilityResult
            Fold evidence, confirmed changes, and descriptive regimes for one
            series.
        """
        boundaries = [0, *(change.position for change in evidence), len(values)]
        regimes = [
            self._regime(values, times, start, end)
            for start, end in pairwise(boundaries)
        ]
        changes = []
        for change, previous, current in zip(
            evidence,
            regimes[:-1],
            regimes[1:],
            strict=True,
        ):
            scale_floor = np.sqrt(np.finfo(float).eps) * max(1.0, abs(previous.mean))
            changes.append(
                TemporalChange(
                    estimated_change_time=times[change.position],
                    detected_at=change.detected_at,
                    distance_ratio=change.distance / change.threshold,
                    diff_mean=current.mean - previous.mean,
                    scale_ratio=(current.std + scale_floor)
                    / (previous.std + scale_floor),
                )
            )
        return TemporalStabilityResult(changes, regimes, windows)

    def _fit_single(
        self, values: np.ndarray, times: np.ndarray | None = None
    ) -> TemporalStabilityResult:
        """Run the complete stability analysis for one ordered series.

        Parameters
        ----------
        values : array-like
            Numeric observations in temporal order.
        times : array-like, optional
            Labels aligned with ``values``. Positional integer labels are used
            when omitted.

        Returns
        -------
        TemporalStabilityResult
            Sequential fold evidence, confirmed changes, and regime summaries.

        Raises
        ------
        ValueError
            If inputs are not finite one-dimensional aligned arrays or do not
            contain enough observations for the initial reference and one fold.
        """
        values, times = self._prepare_series(values, times)
        evidence, windows = self._scan_windows(values, times)
        return self._build_result(values, times, evidence, windows)

    def fit(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> "TemporalStabilityAnalyzer":
        """Analyze every series in a long-format panel.

        The analysis is independent by ``id_col`` and each result is stored in
        ``results_`` under its series identifier. Rows are consumed in their
        existing order; no internal temporal sorting is performed.

        Parameters
        ----------
        df : pandas.DataFrame
            Panel containing identifier, time, and numeric target columns.
        id_col : str, default="unique_id"
            Column identifying independent series.
        time_col : str, default="ds"
            Column providing the time labels retained in result tables.
        target_col : str, default="y"
            Numeric column analyzed for distribution changes.

        Returns
        -------
        TemporalStabilityAnalyzer
            The fitted analyzer.

        Raises
        ------
        TypeError
            If ``df`` is not a pandas DataFrame.
        ValueError
            If panel columns or values are invalid, ID-time pairs are
            duplicated, or a series is too short for the configured initial
            reference and one evaluation fold.

        Notes
        -----
        Observations must already be ordered by ``time_col`` within each
        ``id_col``. The analyzer intentionally preserves the input order.

        Each series must contain at least
        ``horizon * (min_reference_windows + 1)`` observations.
        """
        self._validate_panel(df, id_col, time_col, target_col)
        self._validate_target(df, target_col)
        self.id_col_ = id_col
        self.time_col_ = time_col
        self.target_col_ = target_col
        self.results_ = {
            unique_id: self._fit_single(
                group[target_col].to_numpy(dtype=float),
                group[time_col].to_numpy(),
            )
            for unique_id, group in df.groupby(id_col, sort=False, observed=True)
        }
        return self

    def _require_fitted(self) -> None:
        """Reject result access before a panel has been analyzed."""
        if not hasattr(self, "results_"):
            raise RuntimeError("The analyzer must be fitted before requesting results.")

    def summary(self) -> pd.DataFrame:
        """Return the complete sequence of evaluated folds.

        Each row identifies the evaluated period through ``start_time`` and
        ``end_time``. ``distance_ratio`` is the standardized Wasserstein
        distance divided by its robust threshold: values above one exceed the
        limit. ``status`` is ``stable``, ``candidate``, or ``confirmed``.
        ``reference_size`` reports how many observations supported the
        comparison.

        Returns
        -------
        pandas.DataFrame
            Sequential fold evidence for every fitted series.

        Columns
        -------
        **id_col** : ``object``
            Series identifier using the column name supplied to :meth:`fit`.
        **start_time** : ``object``
            Inclusive time label at the beginning of the evaluated fold.
        **end_time** : ``object``
            Inclusive time label at the end of the evaluated fold.
        **reference_size** : ``int``
            Number of observations supporting the current reference regime.
        **distance_ratio** : ``float``
            Wasserstein distance divided by its calibrated threshold.
        **status** : ``str``
            Fold state: ``"stable"``, ``"candidate"``, or ``"confirmed"``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        self._require_fitted()
        return pd.DataFrame(
            [
                {self.id_col_: unique_id, **window}
                for unique_id, result in self.results_.items()
                for window in result.windows
            ]
        )

    def changes(self) -> pd.DataFrame:
        """Return one row per confirmed distribution change.

        ``estimated_change_time`` is the beginning of the first divergent
        fold, while ``detected_at`` includes the confirmation delay.
        ``distance_ratio`` measures detection evidence relative to the robust
        threshold. ``diff_mean`` and ``scale_ratio`` compare the complete new
        regime with its complete predecessor.

        Returns
        -------
        pandas.DataFrame
            Confirmed changes ordered within each fitted series.

        Columns
        -------
        **id_col** : ``object``
            Series identifier using the column name supplied to :meth:`fit`.
        **change** : ``int``
            One-based change sequence number within the series.
        **estimated_change_time** : ``object``
            Estimated first time label of the new regime.
        **detected_at** : ``object``
            Time label at which persistence confirmed the change.
        **distance_ratio** : ``float``
            Detection distance divided by its calibrated threshold.
        **diff_mean** : ``float``
            New-regime mean minus previous-regime mean.
        **scale_ratio** : ``float``
            New-regime standard deviation divided by previous-regime standard
            deviation, with a numerical floor for degenerate scales.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        self._require_fitted()
        return pd.DataFrame(
            [
                {
                    self.id_col_: unique_id,
                    "change": index,
                    **change.__dict__,
                }
                for unique_id, result in self.results_.items()
                for index, change in enumerate(result.changes, start=1)
            ]
        )

    def regimes(self) -> pd.DataFrame:
        """Return descriptive statistics for every detected regime.

        Regimes are the complete segments delimited by confirmed changes.
        Their bounds are inclusive time labels; ``n_obs`` is the
        number of rows in the segment, and ``mean`` and ``std`` are population
        statistics computed with ``ddof=0``.

        Returns
        -------
        pandas.DataFrame
            Ordered regime segments for every fitted series.

        Columns
        -------
        **id_col** : ``object``
            Series identifier using the column name supplied to :meth:`fit`.
        **regime** : ``int``
            One-based regime sequence number within the series.
        **start_time** : ``object``
            Inclusive first time label of the regime.
        **end_time** : ``object``
            Inclusive final time label of the regime.
        **n_obs** : ``int``
            Number of observations in the regime.
        **mean** : ``float``
            Population mean of the regime.
        **std** : ``float``
            Population standard deviation of the regime using ``ddof=0``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        self._require_fitted()
        return pd.DataFrame(
            [
                {
                    self.id_col_: unique_id,
                    "regime": index,
                    **regime.__dict__,
                }
                for unique_id, result in self.results_.items()
                for index, regime in enumerate(result.regimes, start=1)
            ]
        )
