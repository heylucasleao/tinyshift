"""Decision policies built on row-aligned predictive distributions."""

from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from .distribution import DiscretePredictiveDistribution, PredictiveDistribution

CostInput = str | Real | Mapping[Any | tuple[Any, Any], Real]


class NewsvendorOptimizer:
    """Apply single-period Newsvendor policies to panel forecasts.

    Costs may be numeric scalars, column names, mappings keyed by series ID,
    or mappings keyed by ``(series ID, timestamp)``. Column names are resolved
    from ``cost_df`` when supplied and from ``forecast_df`` otherwise.
    """

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
        """Return the critical-fractile decision for every forecast row.

        Parameters
        ----------
        forecast_df : pandas.DataFrame
            Forecast panel whose row order matches ``distribution``.
        distribution : PredictiveDistribution
            One predictive distribution per row of ``forecast_df``.
        underage_cost, overage_cost : float, str, or mapping
            Non-negative shortage and excess costs. Each value may be a scalar,
            a column name, a mapping keyed by series ID, or a mapping keyed by
            ``(series ID, timestamp)``.
        cost_df : pandas.DataFrame or None, default=None
            Optional source for cost columns. When it contains ``id_col`` and
            ``time_col``, rows are aligned by those keys; otherwise it must
            already have the same row count and order as ``forecast_df``.
        id_col, time_col : str
            Series-ID and timestamp column names used to align costs.
        ratio_col : str, default='critical_ratio'
            Name of the appended critical-ratio column.
        output_col : str, default='y_optimal'
            Name of the appended decision column.

        Returns
        -------
        pandas.DataFrame
            A copy of ``forecast_df`` with the critical ratio and corresponding
            predictive quantile appended.

        Raises
        ------
        TypeError
            If a cost input has an unsupported type.
        ValueError
            If rows or costs cannot be aligned, or costs are negative or
            non-finite.
        """
        cls._validate_alignment(forecast_df, distribution)
        cu, co = cls._resolve_costs(
            forecast_df,
            underage_cost,
            overage_cost,
            cost_df,
            id_col,
            time_col,
        )
        ratio = cls._critical_ratio(cu, co)

        result = forecast_df.copy()
        result[ratio_col] = ratio
        result[output_col] = distribution.ppf(ratio[:, None])
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
        """Return the expected net benefit of each additional inventory unit.

        For unit ``k``, the appended value is the benefit of increasing stock
        from ``k - 1`` to ``k``. This operation is defined only for discrete
        predictive distributions.

        Parameters
        ----------
        forecast_df : pandas.DataFrame
            Forecast panel whose row order matches ``distribution``.
        distribution : PredictiveDistribution
            Discrete predictive distribution for each forecast row.
        underage_cost, overage_cost : float, str, or mapping
            Non-negative shortage and excess costs. Accepted forms are scalar,
            column name, series-ID mapping, and series-timestamp mapping.
        max_k : int or None, default=None
            Largest non-negative unit to evaluate. Produces columns for every
            integer from zero through ``max_k``. Defaults to 10 when neither
            ``max_k`` nor ``units`` is supplied.
        units : iterable of int or None, default=None
            Explicit, unique non-negative units to evaluate. Mutually exclusive
            with ``max_k``; input order determines output-column order.
        cost_df : pandas.DataFrame or None, default=None
            Optional source for cost columns, aligned by ``id_col`` and
            ``time_col`` when both keys are present, or by row otherwise.
        id_col, time_col : str
            Series-ID and timestamp column names used to align costs.

        Returns
        -------
        pandas.DataFrame
            A copy of ``forecast_df`` with one ``MB(k=<unit>)`` column per
            requested unit.

        Raises
        ------
        TypeError
            If ``distribution`` is continuous or a cost input is unsupported.
        ValueError
            If rows or costs cannot be aligned, costs are invalid, or the unit
            specification is invalid.
        """
        cls._validate_alignment(forecast_df, distribution)
        if not isinstance(distribution, DiscretePredictiveDistribution):
            raise TypeError(
                "marginal_benefit is available only for discrete distributions."
            )
        unit_values = cls._resolve_units(max_k, units)
        cu, co = cls._resolve_costs(
            forecast_df,
            underage_cost,
            overage_cost,
            cost_df,
            id_col,
            time_col,
        )
        values = cls._marginal_values(distribution, unit_values, cu, co)
        return cls._append_marginal_columns(forecast_df, unit_values, values)

    @classmethod
    def _resolve_costs(
        cls,
        forecast_df: pd.DataFrame,
        underage_cost: CostInput,
        overage_cost: CostInput,
        cost_df: pd.DataFrame | None,
        id_col: str,
        time_col: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve underage and overage costs on the forecast row grid."""
        costs = cls._align_cost_frame(forecast_df, cost_df, id_col, time_col)
        n_rows = len(forecast_df)
        return (
            cls._extract_cost_array(costs, underage_cost, id_col, time_col, n_rows),
            cls._extract_cost_array(costs, overage_cost, id_col, time_col, n_rows),
        )

    @staticmethod
    def _marginal_values(
        distribution: DiscretePredictiveDistribution,
        units: np.ndarray,
        cu: np.ndarray,
        co: np.ndarray,
    ) -> np.ndarray:
        """Calculate the net benefit of each additional unit."""
        thresholds = np.broadcast_to(units - 1, (len(distribution), len(units)))
        probability_below = np.asarray(distribution.cdf(thresholds))
        return cu[:, None] * (1.0 - probability_below) - co[:, None] * probability_below

    @staticmethod
    def _append_marginal_columns(
        forecast_df: pd.DataFrame, units: np.ndarray, values: np.ndarray
    ) -> pd.DataFrame:
        """Append marginal-benefit columns to a forecast copy."""
        result = forecast_df.copy()
        for index, unit in enumerate(units):
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
            values = NewsvendorOptimizer._mapping_costs(
                frame, cost, id_col, time_col, n_rows
            )
        else:
            raise TypeError(
                "Cost input must be a column name, numeric scalar, or mapping."
            )
        if values.shape != (n_rows,):
            raise ValueError("Cost inputs must have one row per forecast row.")
        return values

    @staticmethod
    def _mapping_costs(
        frame: pd.DataFrame,
        cost: Mapping,
        id_col: str,
        time_col: str,
        n_rows: int,
    ) -> np.ndarray:
        """Resolve series or series-time cost mappings."""
        if not cost:
            raise ValueError("Cost dictionary cannot be empty.")
        if not isinstance(next(iter(cost)), tuple):
            return frame[id_col].map(cost).to_numpy(dtype=float)
        keys = zip(frame[id_col].to_numpy(), frame[time_col].to_numpy())
        return np.fromiter(
            (cost.get(key, np.nan) for key in keys), dtype=float, count=n_rows
        )
