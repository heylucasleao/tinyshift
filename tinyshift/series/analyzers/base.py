"""Shared lifecycle for panel-oriented time-series analyzers."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

import pandas as pd

AnalyzerT = TypeVar("AnalyzerT", bound="BaseSeriesAnalyzer")


class BaseSeriesAnalyzer(ABC):
    """Abstract lifecycle for analyzers operating on panel time series.

    Subclasses implement the analysis of one ordered target series through
    :meth:`_fit_single` and expose a compact tabular view through
    :meth:`summary`. The shared :meth:`fit` implementation validates the panel,
    sorts observations by identifier and time, and stores one result per ID.

    Attributes
    ----------
    results_ : dict
        Results keyed by series identifier. Created by :meth:`fit`; value types
        are defined by each analyzer.
    id_col_, time_col_, target_col_ : str
        Column names used by the most recent call to :meth:`fit`.

    Notes
    -----
    The base class validates panel structure only. Target-domain constraints,
    such as non-negativity or missing-value handling, belong to subclasses via
    :meth:`_validate_target` or :meth:`_fit_single`.
    """

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
        """Fit the analyzer independently to every series in a panel.

        Parameters
        ----------
        df : pandas.DataFrame
            Long-format panel with identifier, time, and target columns.
        id_col : str, default="unique_id"
            Column identifying independent series.
        time_col : str, default="ds"
            Column defining observation order within each series.
        target_col : str, default="y"
            Column containing values analyzed by the subclass.

        Returns
        -------
        BaseSeriesAnalyzer
            The fitted subclass instance.

        Raises
        ------
        TypeError
            If ``df`` is not a pandas DataFrame.
        ValueError
            If required columns are missing, the panel is empty, identifiers
            or times are missing, ID-time pairs are duplicated, or subclass
            target validation fails.
        """
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
