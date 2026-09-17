# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

"""Panel adapters for vector-based drift detectors."""

from typing import Generic, TypeVar

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from .base import BaseDrift
from .categorical import CatDrift
from .continuous import ConDrift

DetectorT = TypeVar("DetectorT", bound=BaseDrift)


class BaseDriftAnalyzer(BaseEstimator, Generic[DetectorT]):
    """Fit one independent drift detector for every panel ID."""

    detector_type: type[BaseDrift]

    def __init__(self, detector: DetectorT) -> None:
        if not isinstance(detector, self.detector_type):
            raise TypeError(
                f"detector must be an instance of {self.detector_type.__name__}."
            )
        self.detector = detector

    @staticmethod
    def _validate_frame(df: pd.DataFrame, id_col: str, target_col: str) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        missing = [column for column in (id_col, target_col) if column not in df]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}.")
        if df.empty:
            raise ValueError("Panel input cannot be empty.")
        if df[id_col].isna().any():
            raise ValueError("ID values must not be missing.")

    def fit(
        self, df: pd.DataFrame, id_col: str = "unique_id", target_col: str = "y"
    ) -> "BaseDriftAnalyzer[DetectorT]":
        """Fit a cloned detector to each ID's reference observations."""
        self._validate_frame(df, id_col, target_col)
        self.id_col_ = id_col
        self.target_col_ = target_col
        self.detectors_: dict[object, DetectorT] = {}
        for unique_id, group in df.groupby(id_col, sort=False, observed=True):
            self.detectors_[unique_id] = clone(self.detector).fit(group[target_col])
        return self

    def predict(
        self, df: pd.DataFrame, id_col: str | None = None, target_col: str | None = None
    ) -> pd.DataFrame:
        """Compare each current ID with its independently fitted reference."""
        check_is_fitted(self, "detectors_")
        id_col = self.id_col_ if id_col is None else id_col
        target_col = self.target_col_ if target_col is None else target_col
        self._validate_frame(df, id_col, target_col)
        unknown = [
            value for value in pd.unique(df[id_col]) if value not in self.detectors_
        ]
        if unknown:
            raise ValueError(f"No reference distribution for IDs: {unknown!r}.")

        rows = []
        for unique_id, group in df.groupby(id_col, sort=False, observed=True):
            result = self.detectors_[unique_id].predict(group[target_col])
            rows.append(
                {
                    id_col: unique_id,
                    "score": result.score,
                    "threshold": result.threshold,
                    "p_value": result.p_value,
                    "drift": result.drift,
                    "reference_size": result.reference_size,
                    "current_size": result.current_size,
                }
            )
        self.results_ = pd.DataFrame(rows)
        return self.results_.copy()

    def summary(self) -> pd.DataFrame:
        """Return a copy of the most recent prediction result."""
        check_is_fitted(self, "results_")
        return self.results_.copy()


class ContinuousDriftAnalyzer(BaseDriftAnalyzer[ConDrift]):
    """Apply :class:`ConDrift` independently to each panel ID."""

    detector_type = ConDrift

    def __init__(self, detector: ConDrift | None = None) -> None:
        super().__init__(ConDrift() if detector is None else detector)


class CategoricalDriftAnalyzer(BaseDriftAnalyzer[CatDrift]):
    """Apply :class:`CatDrift` independently to each panel ID."""

    detector_type = CatDrift

    def __init__(self, detector: CatDrift | None = None) -> None:
        super().__init__(CatDrift() if detector is None else detector)
