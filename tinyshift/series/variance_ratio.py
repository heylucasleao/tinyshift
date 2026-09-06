# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Any, List, Optional

import numpy as np
import pandas as pd

from .diagnostic import variance_ratio


class VarianceRatioAnalyzer:
    """Analyze serial dependence across horizons for panel time series.

    Parameters
    ----------
    horizons : list of int, optional
        Aggregation horizons to evaluate. When omitted, powers of two are
        generated independently for each series, up to the smaller of one
        tenth of its length and ``max_horizon``.
    max_horizon : int, default=64
        Upper bound used only when generating horizons automatically.

    Attributes
    ----------
    results_ : dict
        Mapping from each unique ID to a mapping keyed by horizon. Each result
        contains ``variance_ratio``, ``z_statistic``, and ``p_value``.
    """

    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        max_horizon: int = 64,
    ) -> None:
        self.horizons = horizons
        self.max_horizon = max_horizon
        self._validate_params()

    def __repr__(self) -> str:
        return (
            "VarianceRatioAnalyzer("
            f"horizons={self.horizons!r}, "
            f"max_horizon={self.max_horizon}"
            ")"
        )

    def _validate_params(self) -> None:
        if (
            isinstance(self.max_horizon, bool)
            or not isinstance(self.max_horizon, int)
            or self.max_horizon <= 1
        ):
            raise ValueError("'max_horizon' must be an integer greater than 1.")

        if self.horizons is None:
            return

        if not isinstance(self.horizons, (list, tuple)) or not self.horizons:
            raise ValueError("'horizons' must be a non-empty list of integers.")
        if any(
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or horizon <= 1
            for horizon in self.horizons
        ):
            raise ValueError("Every horizon must be an integer greater than 1.")

    def _default_horizons(self, n: int) -> List[int]:
        """Generate power-of-two horizons supported by the sample size."""
        max_horizon = min(n // 10, self.max_horizon)
        return [
            1 << exponent
            for exponent in range(1, max_horizon.bit_length())
        ]

    def _resolve_horizons(self, n: int) -> List[int]:
        horizons = (
            self._default_horizons(n)
            if self.horizons is None
            else sorted(set(self.horizons))
        )
        invalid = [horizon for horizon in horizons if horizon >= n - 1]
        if invalid:
            raise ValueError(
                "Horizons must be smaller than the number of one-period "
                f"increments; invalid horizons for series length {n}: {invalid}."
            )
        return horizons

    def _fit_single(self, values: np.ndarray) -> dict[int, dict[str, float]]:
        horizons = self._resolve_horizons(len(values))
        results: dict[int, dict[str, float]] = {}

        for horizon in horizons:
            try:
                ratio, z_statistic, p_value = variance_ratio(values, horizon=horizon)
            except ValueError as error:
                if "zero variance" not in str(error):
                    raise
                ratio = z_statistic = p_value = float("nan")

            results[horizon] = {
                "variance_ratio": ratio,
                "z_statistic": z_statistic,
                "p_value": p_value,
            }

        return results

    def fit(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> "VarianceRatioAnalyzer":
        """Fit the analyzer to each series in a panel DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame in panel format.")

        required = [id_col, time_col, target_col]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}.")
        if df.empty:
            raise ValueError("Panel input must contain at least one series.")
        if df[required].isna().any().any():
            raise ValueError("ID, time, and target values must not be missing.")
        if df.duplicated([id_col, time_col]).any():
            raise ValueError("Panel contains duplicate ID-time observations.")
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise ValueError(f"Target column {target_col!r} must be numeric.")
        if not np.isfinite(df[target_col].to_numpy(dtype=float)).all():
            raise ValueError("Target values must be finite.")

        self.id_col_ = id_col
        self.time_col_ = time_col
        self.target_col_ = target_col

        ordered = df.sort_values([id_col, time_col])
        self.results_ = {
            unique_id: self._fit_single(group[target_col].to_numpy(dtype=float))
            for unique_id, group in ordered.groupby(id_col, sort=False, observed=True)
        }
        return self

    def profile(self) -> pd.DataFrame:
        """Return one row per series and evaluated horizon."""
        if not hasattr(self, "results_"):
            raise RuntimeError(
                "The analyzer must be fitted before calling `profile()`."
            )

        rows = [
            {
                self.id_col_: unique_id,
                "horizon": horizon,
                **result,
            }
            for unique_id, horizon_results in self.results_.items()
            for horizon, result in horizon_results.items()
        ]
        columns = [
            self.id_col_,
            "horizon",
            "variance_ratio",
            "z_statistic",
            "p_value",
        ]
        return pd.DataFrame(rows, columns=columns)
