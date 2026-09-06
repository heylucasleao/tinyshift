"""Shared lifecycle for panel-oriented time-series analyzers."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

import pandas as pd

AnalyzerT = TypeVar("AnalyzerT", bound="BaseSeriesAnalyzer")


class BaseSeriesAnalyzer(ABC):
    """Base class implementing validation, ordering, and panel iteration."""

    @staticmethod
    def _validate_panel(
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame in panel format.")
        required = [id_col, time_col, target_col]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}.")
        if df.empty:
            raise ValueError("Panel input must contain at least one series.")
        if df[[id_col, time_col]].isna().any().any():
            raise ValueError("ID and time values must not be missing.")
        if df.duplicated([id_col, time_col]).any():
            raise ValueError("Panel contains duplicate ID-time observations.")

    def _validate_target(self, df: pd.DataFrame, target_col: str) -> None:
        """Validate target values before grouping; subclasses may override."""

    @abstractmethod
    def _fit_single(self, values: pd.Series) -> Any:
        """Analyze one time-ordered series."""

    def fit(
        self: AnalyzerT,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> AnalyzerT:
        """Analyze every series in a panel DataFrame."""
        self._validate_panel(df, id_col, time_col, target_col)
        self._validate_target(df, target_col)
        self.id_col_ = id_col
        self.time_col_ = time_col
        self.target_col_ = target_col

        ordered = df.sort_values([id_col, time_col])
        self.results_ = {
            unique_id: self._fit_single(group[target_col])
            for unique_id, group in ordered.groupby(
                id_col, sort=False, observed=True
            )
        }
        return self

    @abstractmethod
    def summary(self) -> pd.DataFrame:
        """Return the analyzer's compact panel result."""
