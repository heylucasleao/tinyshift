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

    index: int
    start_time: Any
    end_time: Any
    n_observations: int
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
    """

    def __init__(
        self,
        horizon: int,
        *,
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
    def _reference_scale(values: np.ndarray) -> tuple[float, float]:
        """Return reference standard deviation and its stabilized denominator."""
        standard_deviation = float(np.std(values, ddof=0))
        floor = np.sqrt(np.finfo(float).eps) * max(1.0, abs(float(np.median(values))))
        return standard_deviation, max(standard_deviation, floor)

    def _distance(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Return Wasserstein distance standardized by reference dispersion."""
        _, denominator = self._reference_scale(reference)
        return float(wasserstein_distance(reference, current) / denominator)

    def _resolve_threshold(self, reference: np.ndarray) -> float:
        """Calibrate a robust limit from sequential historical pseudo-folds."""
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

    def _fold_starts(self, n_observations: int) -> list[int]:
        """Return eligible fold starts under the configured temporal geometry."""
        starts = list(
            range(
                self.min_reference_size_,
                n_observations - self.horizon + 1,
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
        index: int,
        start: int,
        end: int,
    ) -> TemporalRegime:
        """Summarize one half-open segment as a temporal regime."""
        segment = values[start:end]
        return TemporalRegime(
            index=index,
            start_time=times[start],
            end_time=times[end - 1],
            n_observations=len(segment),
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
        """Scan folds sequentially and retain confirmed change evidence."""
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
            if start - regime_start < self.min_reference_size_:
                continue
            reference_end = pending_start if pending_start is not None else start
            reference = values[regime_start:reference_end]
            current = values[start : start + self.horizon]
            limit = self._resolve_threshold(reference)
            distance = self._distance(reference, current)
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
        """Build regime summaries and changes from confirmed evidence."""
        boundaries = [0, *(change.position for change in evidence), len(values)]
        regimes = [
            self._regime(values, times, index, start, end)
            for index, (start, end) in enumerate(pairwise(boundaries))
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
        """Orchestrate the stability analysis of one ordered series."""
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
        """Fit each panel series in its existing row order.

        Observations must already be ordered by ``time_col`` within each
        ``id_col``. The analyzer intentionally does not reorder the input.
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
        """Return every sequential reference-versus-fold comparison."""
        self._require_fitted()
        return pd.DataFrame(
            [
                {self.id_col_: unique_id, **window}
                for unique_id, result in self.results_.items()
                for window in result.windows
            ]
        )

    def changes(self) -> pd.DataFrame:
        """Return one row per confirmed distribution change."""
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
        """Return one descriptive row per detected temporal regime."""
        self._require_fitted()
        return pd.DataFrame(
            [
                {self.id_col_: unique_id, "regime": regime.index, **regime.__dict__}
                for unique_id, result in self.results_.items()
                for regime in result.regimes
            ]
        ).drop(columns="index", errors="ignore")
