"""Decision policies built on row-aligned predictive distributions."""

from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from .distribution import DiscretePredictiveDistribution, PredictiveDistribution

CostInput = str | Real | Mapping[Any | tuple[Any, Any], Real]


class NewsvendorOptimizer:
    """Apply single-period Newsvendor decisions to predictive distributions."""

    @classmethod
    def optimize(
        cls,
        forecast_df: pd.DataFrame,
        distribution: PredictiveDistribution,
        underage_cost: CostInput,
        overage_cost: CostInput,
        cost_df: pd.DataFrame | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        ratio_col: str = "critical_ratio",
        output_col: str = "y_optimal",
    ) -> pd.DataFrame:
        """Return the critical-fractile decision for every forecast row."""
        cls._validate_alignment(forecast_df, distribution)
        costs = cls._align_cost_frame(forecast_df, cost_df, id_col, time_col)
        cu = cls._extract_cost_array(
            costs, underage_cost, id_col, time_col, len(forecast_df)
        )
        co = cls._extract_cost_array(
            costs, overage_cost, id_col, time_col, len(forecast_df)
        )
        ratio = cls._critical_ratio(cu, co)

        result = forecast_df.copy()
        result[ratio_col] = ratio
        result[output_col] = distribution.ppf(ratio)
        return result

    @classmethod
    def marginal_benefit(
        cls,
        forecast_df: pd.DataFrame,
        distribution: PredictiveDistribution,
        underage_cost: CostInput,
        overage_cost: CostInput,
        max_k: int | None = None,
        units: Iterable[int] | None = None,
        cost_df: pd.DataFrame | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
    ) -> pd.DataFrame:
        """Return the expected net benefit of each additional inventory unit."""
        cls._validate_alignment(forecast_df, distribution)
        if not isinstance(distribution, DiscretePredictiveDistribution):
            raise TypeError(
                "marginal_benefit is available only for discrete distributions."
            )
        unit_values = cls._resolve_units(max_k, units)
        costs = cls._align_cost_frame(forecast_df, cost_df, id_col, time_col)
        cu = cls._extract_cost_array(
            costs, underage_cost, id_col, time_col, len(forecast_df)
        )
        co = cls._extract_cost_array(
            costs, overage_cost, id_col, time_col, len(forecast_df)
        )
        thresholds = np.broadcast_to(
            unit_values - 1, (len(forecast_df), len(unit_values))
        )
        probability_below = np.asarray(distribution.cdf(thresholds))
        values = (
            cu[:, None] * (1.0 - probability_below) - co[:, None] * probability_below
        )

        result = forecast_df.copy()
        for index, unit in enumerate(unit_values):
            result[f"MB(k={unit})"] = values[:, index]
        return result

    @staticmethod
    def _validate_alignment(
        forecast_df: pd.DataFrame, distribution: PredictiveDistribution
    ) -> None:
        if len(forecast_df) != len(distribution):
            raise ValueError(
                "forecast_df and distribution must contain the same number of rows."
            )

    @staticmethod
    def _critical_ratio(cu: np.ndarray, co: np.ndarray) -> np.ndarray:
        if np.any(~np.isfinite(cu)) or np.any(~np.isfinite(co)):
            raise ValueError("Costs must be finite.")
        if np.any(cu < 0) or np.any(co < 0):
            raise ValueError("Costs must be non-negative.")
        denominator = cu + co
        result = np.full_like(denominator, 0.5)
        np.divide(cu, denominator, out=result, where=denominator != 0)
        return result

    @staticmethod
    def _resolve_units(max_k: int | None, units: Iterable[int] | None) -> np.ndarray:
        if max_k is not None and units is not None:
            raise ValueError("Provide either max_k or units, not both.")
        if units is None:
            max_k = 10 if max_k is None else max_k
            if (
                isinstance(max_k, (bool, np.bool_))
                or not isinstance(max_k, (int, np.integer))
                or max_k < 0
            ):
                raise ValueError("max_k must be a non-negative integer.")
            return np.arange(max_k + 1)
        if isinstance(units, (str, bytes)):
            raise ValueError(  # noqa: TRY004
                "units must be a non-empty iterable of integers."
            )
        try:
            values = list(units)
        except TypeError as exc:
            raise ValueError("units must be a non-empty iterable of integers.") from exc
        if not values or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 0
            for value in values
        ):
            raise ValueError(
                "units must be a non-empty iterable of non-negative integers."
            )
        if len(set(values)) != len(values):
            raise ValueError("units must not contain duplicates.")
        return np.asarray(values, dtype=int)

    @staticmethod
    def _align_cost_frame(
        forecast_df: pd.DataFrame,
        cost_df: pd.DataFrame | None,
        id_col: str,
        time_col: str,
    ) -> pd.DataFrame:
        if cost_df is None:
            return forecast_df
        if not {id_col, time_col}.issubset(cost_df.columns):
            if len(cost_df) != len(forecast_df):
                raise ValueError("Cost inputs must have one row per forecast row.")
            return cost_df
        return forecast_df[[id_col, time_col]].merge(
            cost_df,
            on=[id_col, time_col],
            how="left",
            sort=False,
            validate="one_to_one",
        )

    @staticmethod
    def _extract_cost_array(
        frame: pd.DataFrame,
        cost: CostInput,
        id_col: str,
        time_col: str,
        n_rows: int,
    ) -> np.ndarray:
        if isinstance(cost, (bool, np.bool_)):
            raise TypeError("Cost inputs cannot be boolean.")
        if isinstance(cost, Real):
            values = np.full(n_rows, float(cost), dtype=float)
        elif isinstance(cost, str):
            if cost not in frame.columns:
                raise ValueError(f"Cost column not found: {cost!r}.")
            values = frame[cost].to_numpy(dtype=float)
        elif isinstance(cost, Mapping):
            if not cost:
                raise ValueError("Cost dictionary cannot be empty.")
            first_key = next(iter(cost))
            if isinstance(first_key, tuple):
                keys = zip(frame[id_col].to_numpy(), frame[time_col].to_numpy())
                values = np.fromiter(
                    (cost.get(key, np.nan) for key in keys),
                    dtype=float,
                    count=n_rows,
                )
            else:
                values = frame[id_col].map(cost).to_numpy(dtype=float)
        else:
            raise TypeError(
                "Cost input must be a column name, numeric scalar, or mapping."
            )
        if values.shape != (n_rows,):
            raise ValueError("Cost inputs must have one row per forecast row.")
        return values
