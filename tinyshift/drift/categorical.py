# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.base import BaseEstimator

from .base import BaseModel


def chebyshev(a, b):
    """
    Compute the Chebyshev distance between two distributions.
    """
    return np.max(np.abs(a - b))


def psi(observed, expected, epsilon=1e-4):
    """
    Calculate Population Stability Index (PSI) between two distributions.
    """
    observed = np.clip(observed, epsilon, 1)
    expected = np.clip(expected, epsilon, 1)
    return np.sum((observed - expected) * np.log(observed / expected))


class CatDrift(BaseModel, BaseEstimator):
    """
    A tracker for identifying drift in categorical data over time.

    The tracker uses a reference dataset to compute a baseline distribution and compares
    subsequent data for deviations based on a distance metric and drift limits.

    Available distance metrics:
    - 'chebyshev': Maximum absolute difference between category probabilities
    - 'jensenshannon': Jensen-Shannon divergence (symmetric, sqrt of JS distance)
    - 'psi': Population Stability Index (sensitive to small probability changes)

    Attributes
    ----------
    func : Callable
        The distance function used for drift calculation.
    reference_distribution : dict
        Normalized probability distribution of reference categories.
    method : str
        The comparison method being used.
    freq : str
        The frequency parameter for time grouping.
    """

    def __init__(
        self,
        freq: str | None = None,
        func: str = "chebyshev",
        drift_limit: str | tuple[float | None, float | None] = "auto",
        method: str = "expanding",
    ):
        """
        Initialize the categorical drift detector.

        Parameters
        ----------
        freq : str
            Frequency for time grouping (e.g., 'D', 'W', 'M'). Required for time-based analysis.
        func : str, default='chebyshev'
            Distance metric to use for drift detection. Options: 'chebyshev', 'jensenshannon', 'psi'.
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
    ) -> "CatDrift":
        """
        Fit the drift detector to reference data.

        Parameters
        ----------
        df : pd.DataFrame
            Reference dataframe containing categorical data with time series structure.
        id_col : str, default='unique_id'
            Column name identifying unique time series entities.
        time_col : str, default='ds'
            Column name containing timestamps for time-based grouping.
        target_col : str, default='y'
            Column name containing categorical values to analyze for drift.

        Returns
        -------
        self : CatDrift
            Returns self for method chaining.
        """
        self._check_dataframe(df, time_col, target_col, id_col)
        self._distance_function_ = self._selection_function(self.func)

        frequency = (
            df.groupby([id_col, pd.Grouper(key=time_col, freq=self.freq), target_col])[
                target_col
            ]
            .size()
            .unstack(fill_value=0)
        )
        reference_counts = frequency.groupby(level=id_col).sum()
        reference = reference_counts.div(reference_counts.sum(axis=1), axis=0)
        reference_distance = self._reference_distances_by_series(
            frequency, id_col, time_col
        )

        self.reference_distribution = {
            unique_id: {
                category: float(prob)
                for category, prob in reference.loc[unique_id].items()
            }
            for unique_id in reference.index
        }

        super().__init__(
            reference_distance,
            self.drift_limit,
            id_col,
        )

        return self

    def _reference_distances_by_series(
        self, frequency: pd.DataFrame, id_col: str, time_col: str
    ) -> pd.Series:
        """Calibrate temporal distances independently for each series."""
        parts = []
        for unique_id in frequency.index.get_level_values(id_col).unique():
            group = frequency.xs(unique_id, level=id_col)
            distances = self._generate_distance(group)
            distances.index = pd.MultiIndex.from_arrays(
                [[unique_id] * len(distances), distances.index],
                names=[id_col, time_col],
            )
            parts.append(distances)
        return pd.concat(parts)

    def _selection_function(self, func_name: str) -> Callable:
        """Returns a specific function based on the given function name."""

        if func_name == "chebyshev":
            selected_func = chebyshev
        elif func_name == "jensenshannon":
            selected_func = jensenshannon
        elif func_name == "psi":
            selected_func = psi
        else:
            raise ValueError(f"Unsupported distance function: {func_name}")
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
            Frequency counts of categories per period. Rows = time periods,
            columns = categories.

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
        n = len(X)
        if n < 2:
            raise ValueError(
                "Each series requires at least two reference periods for expanding calibration."
            )
        distances = np.zeros(n - 1)

        past_value = np.zeros(X.shape[1], dtype=np.float64)
        for i in range(1, n):
            past_value = past_value + X[i - 1]
            past_value_norm = past_value / np.sum(past_value)
            current_value_norm = X[i] / np.sum(X[i])
            distances[i - 1] = self._distance_function_(
                past_value_norm, current_value_norm
            )

        return pd.Series(distances, index=index[1:])

    def _jackknife_distance(self, X: np.ndarray, index) -> pd.Series:
        """Compute distances using jackknife (leave-one-out) approach."""
        n = len(X)
        if n < 2:
            raise ValueError(
                "Each series requires at least two reference periods for jackknife calibration."
            )
        distances = np.zeros(n)

        for i in range(n):
            current_value_norm = X[i] / np.sum(X[i])
            past_value = np.delete(X, i, axis=0)
            past_value_norm = past_value.sum(axis=0) / np.sum(past_value.sum(axis=0))
            distances[i] = self._distance_function_(past_value_norm, current_value_norm)

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
        if not hasattr(self, "reference_distribution"):
            raise ValueError("Model must be fitted before scoring.")
        period_distributions = self._period_distributions(
            df, id_col, time_col, target_col
        )
        return self._score_period_distributions(
            period_distributions, id_col, time_col
        )

    def _period_distributions(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
    ) -> pd.DataFrame:
        """Build normalized category probabilities per series and period."""
        frequency = (
            df.groupby([id_col, pd.Grouper(key=time_col, freq=self.freq), target_col])[
                target_col
            ]
            .size()
            .unstack(fill_value=0)
        )
        return frequency.div(frequency.sum(axis=1), axis=0)

    @staticmethod
    def _align_categories(
        current: pd.DataFrame, reference: dict
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Align the union of current and reference category supports."""
        categories = list(reference)
        categories.extend(
            category for category in current.columns if category not in reference
        )
        current_aligned = current.reindex(columns=categories, fill_value=0.0)
        reference_aligned = np.array(
            [reference.get(category, 0.0) for category in categories]
        )
        return current_aligned, reference_aligned

    def _score_series(
        self,
        unique_id,
        current: pd.DataFrame,
        id_col: str,
        time_col: str,
    ) -> pd.DataFrame:
        """Compare every period of one series with its reference distribution."""
        if unique_id not in self.reference_distribution:
            raise ValueError(
                f"No reference distribution is available for series: {unique_id!r}."
            )
        current, reference = self._align_categories(
            current, self.reference_distribution[unique_id]
        )
        distances = np.fromiter(
            (self._distance_function_(row, reference) for row in current.values),
            dtype=float,
            count=len(current),
        )
        return pd.DataFrame(
            {
                id_col: unique_id,
                time_col: current.index,
                "metric": distances,
            }
        )

    def _score_period_distributions(
        self, periods: pd.DataFrame, id_col: str, time_col: str
    ) -> pd.DataFrame:
        """Score all categorical series and combine their period results."""
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
