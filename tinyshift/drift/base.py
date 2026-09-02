# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from tinyshift.stats import StatisticalInterval


class BaseModel(ABC):
    def __init__(
        self,
        reference: pd.Series,
        drift_limit: str | tuple[float | None, float | None],
        id_col: str = "unique_id",
    ):
        """
        Initialize the BaseModel class with reference distribution and drift limits.

        Parameters
        ----------
        reference : pd.Series
            Series containing the reference distribution with id_col as grouping variable.
        drift_limit : Union[str, Tuple[float, float]]
            Method for determining drift thresholds ("deviation" or "mad") or custom limits as a tuple.
        id_col : str, default "unique_id"
            Column name used for grouping the reference data.
        """
        self._threshold_cache = (
            reference.groupby(id_col)
            .apply(self._get_drift_threshold, drift_limit)
            .to_dict()
        )
        invalid = {
            unique_id: threshold
            for unique_id, threshold in self._threshold_cache.items()
            if not np.isfinite(threshold)
        }
        if invalid:
            raise ValueError(
                "Drift thresholds must be finite. Provide more reference periods "
                "or an explicit upper drift limit."
            )

    def _check_dataframe(
        self, df: pd.DataFrame, time_col: str, target_col: str, id_col: str
    ):
        """
        Validate the input DataFrame for required columns and types.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        if time_col not in df.columns:
            raise ValueError(f"time_col '{time_col}' not found in DataFrame.")
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in DataFrame.")
        if id_col not in df.columns:
            raise ValueError(f"id_col '{id_col}' not found in DataFrame.")
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            raise ValueError(f"time_col '{time_col}' must be datetime.")
        if df.empty:
            raise ValueError("Input data cannot be empty.")
        if df[[id_col, time_col, target_col]].isna().any().any():
            raise ValueError("ID, time, and target columns cannot contain missing values.")

    def _get_drift_threshold(
        self,
        reference_metrics: pd.Series,
        drift_limit: str | tuple[float | None, float | None],
    ) -> float:
        """
        Helper function to compute drift threshold based on specified method or custom limits.
        """
        _, drift_threshold = StatisticalInterval.compute_interval(
            reference_metrics, drift_limit
        )
        return drift_threshold

    def _get_index(self, X: pd.Series | list[np.ndarray] | list[list]):
        """
        Helper function to retrieve the index of a pandas Series or generate a default index.
        """
        return X.index if hasattr(X, "index") else list(range(len(X)))

    def _is_drifted(self, df: pd.DataFrame, id_col: str) -> pd.Series:
        """
        Vectorized version of drift detection - much faster than transform + lambda.
        """
        unknown = set(df[id_col].unique()) - set(self._threshold_cache)
        if unknown:
            raise ValueError(
                f"No reference distribution is available for series: {sorted(unknown)!r}."
            )
        thresholds = df[id_col].map(self._threshold_cache).to_numpy(dtype=float)
        return pd.Series(df["metric"].to_numpy() > thresholds, index=df.index)

    @property
    def thresholds(self) -> dict[object, float]:
        """Get the drift thresholds for each group as dict for faster access."""
        return self._threshold_cache

    @abstractmethod
    def score(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.DataFrame:
        """
        Compute the drift metric for each time period in the provided dataset.
        """
        raise NotImplementedError

    def predict(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.DataFrame:
        """
        Predict drift for each time period in the dataset compared to the reference.
        """
        metrics = self.score(
            df,
            id_col,
            time_col,
            target_col,
        )

        metrics["drift"] = self._is_drifted(metrics, id_col)
        return metrics
