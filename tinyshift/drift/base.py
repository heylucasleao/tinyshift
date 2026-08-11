# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
from typing import Union, Tuple, List
import pandas as pd
from tinyshift.stats import StatisticalInterval
from abc import ABC, abstractmethod
from typing import Dict


class BaseModel(ABC):
    def __init__(
        self,
        reference: pd.Series,
        drift_limit: Union[str, Tuple[float, float]],
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

    def _check_dataframe(
        self, df: pd.DataFrame, time_col: str, target_col: str, id_col: str
    ):
        """
        Validate the input DataFrame for required columns and types.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        if time_col not in df.columns:
            raise ValueError(f"time_col '{time_col}' not found in DataFrame.")
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in DataFrame.")
        if id_col not in df.columns:
            raise ValueError(f"id_col '{id_col}' not found in DataFrame.")
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            raise ValueError(f"time_col '{time_col}' must be datetime.")

    def _get_drift_threshold(
        self,
        reference_metrics: pd.Series,
        drift_limit: Union[str, Tuple[float, float]],
    ) -> float:
        """
        Helper function to compute drift threshold based on specified method or custom limits.
        """
        _, drift_threshold = StatisticalInterval.compute_interval(
            reference_metrics, drift_limit
        )
        return drift_threshold

    def _get_index(self, X: Union[pd.Series, List[np.ndarray], List[list]]):
        """
        Helper function to retrieve the index of a pandas Series or generate a default index.
        """
        return X.index if hasattr(X, "index") else list(range(len(X)))

    def _is_drifted(self, df: pd.DataFrame, id_col: str) -> pd.Series:
        """
        Vectorized version of drift detection - much faster than transform + lambda.
        """
        mask = np.zeros(len(df), dtype=bool)

        for unique_id, idx in df.groupby(id_col).groups.items():
            threshold = self._threshold_cache[unique_id]
            group_metrics = df.loc[idx, "metric"].values
            mask[idx] = group_metrics >= threshold

        return pd.Series(mask, index=df.index)

    @property
    def thresholds(self) -> Dict[str, float]:
        """Get the drift thresholds for each group as dict for faster access."""
        return self._threshold_cache

    @abstractmethod
    def score(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.Series:
        """
        Compute the drift metric for each time period in the provided dataset.
        """
        pass

    def predict(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.Series:
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
