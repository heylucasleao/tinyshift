# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.base import BaseEstimator

from .base import BaseModel


class ConDrift(BaseModel, BaseEstimator):
    """
    A tracker for identifying drift in continuous data over time.

    The tracker uses a reference dataset to compute a baseline distribution and compares
    subsequent data for deviations based on a distance metric and drift limits.

    Available distance metrics:
    - 'ws': Wasserstein distance (Earth Mover's Distance) - measures the minimum cost
      to transform one distribution into another

    Comparison methods:
    - 'expanding': Each point compared against all accumulated past data
    - 'jackknife': Each point compared against all other points (leave-one-out)

    Attributes
    ----------
    func : Callable
        The distance function used for drift calculation.
    reference_distribution : dict
        Dictionary mapping unique_id to reference data arrays used as baseline.
    method : str
        The comparison method being used.
    freq : str
        The frequency parameter for time grouping.
    """

    def __init__(
        self,
        freq: str | None = None,
        func: str = "ws",
        drift_limit: str | tuple[float | None, float | None] = "auto",
        method: str = "expanding",
    ):
        """
        Initialize the continuous drift detector.

        Parameters
        ----------
        freq : str
            Frequency for time grouping (e.g., 'D', 'W', 'M'). Required for time-based analysis.
        func : str, default='ws'
            Distance metric to use for drift detection. Options: 'ws' (Wasserstein distance).
        drift_limit : Union[str, Tuple[float, float]], default='auto'
            Drift threshold definition. Use 'auto' for automatic thresholds or
            provide custom (lower, upper) bounds.
        method : str, default='expanding'
            Comparison method:
            - 'expanding': Each point compared against accumulated past data
            - 'jackknife': Each point compared against all other points (leave-one-out)
        """

        if freq is None:
            raise ValueError("freq must be specified for time grouping.")

        if method not in ["expanding", "jackknife"]:
            raise ValueError(
                f"method must be one of ['expanding', 'jackknife'], got '{method}'"
            )

        self.freq = freq
        self._selection_function(func)
        self.func = func
        self.drift_limit = drift_limit
        self.method = method

    def fit(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> "ConDrift":
        """
        Fit the drift detector to reference data.

        Parameters
        ----------
        df : pd.DataFrame
            Reference dataframe containing continuous data with time series structure.
        id_col : str, default='unique_id'
            Column name identifying unique time series entities.
        time_col : str, default='ds'
            Column name containing timestamps for time-based grouping.
        target_col : str, default='y'
            Column name containing continuous values to analyze for drift.

        Returns
        -------
        self : ConDrift
            Returns self for method chaining.
        """
        self._check_dataframe(df, time_col, target_col, id_col)
        self._validate_continuous_target(df[target_col], target_col)
        self._distance_function_ = self._selection_function(self.func)

        reference = df.groupby([id_col, pd.Grouper(key=time_col, freq=self.freq)])[
            target_col
        ].apply(np.asarray)

        reference_distance = self._reference_distances_by_series(
            reference, id_col, time_col
        )

        self.reference_distribution = {
            unique_id: np.concatenate(reference.loc[unique_id].values).astype(float)
            for unique_id in reference.index.get_level_values(0).unique()
        }

        super().__init__(
            reference_distance,
            self.drift_limit,
            id_col,
        )

        return self

    @staticmethod
    def _validate_continuous_target(target: pd.Series, target_col: str) -> None:
        """Require finite numeric observations for Wasserstein distance."""
        if not pd.api.types.is_numeric_dtype(target):
            raise ValueError(f"target_col '{target_col}' must be numeric.")
        if not np.isfinite(target.to_numpy(dtype=float)).all():
            raise ValueError(f"target_col '{target_col}' must contain finite values.")

    def _reference_distances_by_series(
        self, reference: pd.Series, id_col: str, time_col: str
    ) -> pd.Series:
        """Calibrate temporal distances independently for each series."""
        parts = []
        for unique_id in reference.index.get_level_values(id_col).unique():
            group = reference.xs(unique_id, level=id_col)
            distances = self._generate_distance(group)
            distances.index = pd.MultiIndex.from_arrays(
                [[unique_id] * len(distances), distances.index],
                names=[id_col, time_col],
            )
            parts.append(distances)
        return pd.concat(parts)

    def _selection_function(self, func_name: str) -> Callable:
        """Returns a specific function based on the given function name."""

        if func_name == "ws":
            selected_func = wasserstein_distance
        else:
            raise ValueError(f"Unsupported function: {func_name}")
        return selected_func

    def _generate_distance(
        self,
        X: pd.Series | list[np.ndarray] | list[list],
    ) -> pd.Series:
        """
        Compute a distance metric using different comparison strategies.

        - **Expanding window (method='expanding')**:
            Each point is compared against all accumulated past data.
            Best for detecting gradual drift over time. Efficient O(n).

        - **Jackknife (method='jackknife')**:
            Each point is compared against all other points (leave-one-out).
            Better for detecting point anomalies. Computationally intensive O(n²).

        Parameters
        ----------
        X : Union[pd.Series, List[np.ndarray], List[list]]
            Input data to compute distances. If Series, uses its index for the output.

        Returns
        -------
        pd.Series
            Distance metrics indexed by time period. Note:
            - Expanding: First period is dropped (no reference)
            - Jackknife: All periods included
        """
        index = self._get_index(X)
        X = np.asarray(X)

        if self.method == "expanding":
            return self._expanding_distance(X, index)
        if self.method == "jackknife":
            return self._jackknife_distance(X, index)
        raise ValueError(f"Unknown method: {self.method}")

    def _expanding_distance(self, X: np.ndarray, index) -> pd.Series:
        """Compute distances using expanding window approach."""
        if X.shape[0] < 2:
            raise ValueError(
                "Each series requires at least two reference periods for expanding calibration."
            )
        distances = np.zeros(X.shape[0] - 1)

        past_value = np.array([], dtype=float)
        for i in range(1, X.shape[0]):
            past_value = np.concatenate([past_value, X[i - 1]])
            distances[i - 1] = self._distance_function_(past_value, X[i])

        return pd.Series(distances, index=index[1:])

    def _jackknife_distance(self, X: np.ndarray, index) -> pd.Series:
        """Compute distances using jackknife (leave-one-out) approach."""
        if X.shape[0] < 2:
            raise ValueError(
                "Each series requires at least two reference periods for jackknife calibration."
            )
        distances = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            past_value = np.concatenate(np.delete(np.asarray(X), i, axis=0))
            distances[i] = self._distance_function_(past_value, X[i])

        return pd.Series(distances, index=index)

    def score(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.DataFrame:
        """
        Compute the drift metric between the reference distribution and new data points.
        """
        self._check_dataframe(df, time_col, target_col, id_col)
        self._validate_continuous_target(df[target_col], target_col)
        if not hasattr(self, "reference_distribution"):
            raise ValueError("Model must be fitted before scoring.")
        periods = self._group_periods(df, id_col, time_col, target_col)
        return self._score_period_groups(periods, id_col, time_col)

    def _group_periods(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
    ) -> pd.Series:
        """Aggregate continuous observations into arrays by series and period."""
        return df.groupby(
            [id_col, pd.Grouper(key=time_col, freq=self.freq)]
        )[target_col].apply(np.asarray)

    def _score_series(
        self,
        unique_id,
        periods: pd.Series,
        id_col: str,
        time_col: str,
    ) -> pd.DataFrame:
        """Compare every period of one series with its reference distribution."""
        if unique_id not in self.reference_distribution:
            raise ValueError(
                f"No reference distribution is available for series: {unique_id!r}."
            )
        reference = self.reference_distribution[unique_id]
        distances = np.fromiter(
            (
                self._distance_function_(current, reference)
                for current in periods.values
            ),
            dtype=float,
            count=len(periods),
        )
        return pd.DataFrame(
            {
                id_col: unique_id,
                time_col: periods.index,
                "metric": distances,
            }
        )

    def _score_period_groups(
        self, periods: pd.Series, id_col: str, time_col: str
    ) -> pd.DataFrame:
        """Score all continuous series and combine their period results."""
        return pd.concat(
            (
                self._score_series(
                    unique_id,
                    periods.loc[unique_id],
                    id_col,
                    time_col,
                )
                for unique_id in periods.index.get_level_values(0).unique()
            ),
            ignore_index=True,
        )
