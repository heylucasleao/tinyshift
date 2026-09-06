"""Linear-trend analysis for panel time series."""

from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from ..diagnostic import trend_significance
from .base import BaseSeriesAnalyzer


class TrendAnalyzer(BaseSeriesAnalyzer):
    """Analyze linear trend magnitude and significance for panel series.

    A least-squares regression of target values on observation order is fitted
    independently per ID. The analyzer reports slope, explained variance, the
    two-sided slope-test p-value, and a thresholded significance flag.

    Parameters
    ----------
    significance_level : float, default=0.05
        P-value threshold used for ``significant_trend``. Must lie strictly
        between zero and one.

    Attributes
    ----------
    results_ : dict
        Mapping from ID to ``trend_slope``, ``trend_r2``, ``trend_pvalue``,
        and ``significant_trend``.

    Notes
    -----
    The slope is expressed in target units per observation. The time column is
    used for ordering, not as a numeric regression covariate; irregularly
    sampled panels should therefore be regularized before analysis.

    Examples
    --------
    >>> analyzer = TrendAnalyzer(significance_level=0.05)
    >>> analyzer.fit(df).summary()
      unique_id  trend_slope  trend_r2  trend_pvalue  significant_trend
    0         A          ...       ...           ...                ...
    """

    def __init__(self, significance_level: float = 0.05) -> None:
        self.significance_level = significance_level
        if (
            isinstance(significance_level, bool)
            or not isinstance(significance_level, Real)
            or not np.isfinite(significance_level)
            or not 0 < significance_level < 1
        ):
            raise ValueError("'significance_level' must be between 0 and 1.")

    def __repr__(self) -> str:
        return f"TrendAnalyzer(significance_level={self.significance_level})"

    def _validate_target(self, df: pd.DataFrame, target_col: str) -> None:
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise ValueError(f"Target column {target_col!r} must be numeric.")
        if not np.isfinite(df[target_col].to_numpy(dtype=float)).all():
            raise ValueError("Target values must be finite.")

    def _fit_single(self, values: pd.Series) -> dict[str, Any]:
        slope, r_squared, p_value = trend_significance(values)
        return {
            "trend_slope": slope,
            "trend_r2": r_squared,
            "trend_pvalue": p_value,
            "significant_trend": bool(p_value < self.significance_level),
        }

    def summary(self) -> pd.DataFrame:
        """Return linear-trend diagnostics with one row per series.

        Returns
        -------
        pandas.DataFrame
            ID plus slope, R², p-value, and significance flag columns.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not hasattr(self, "results_"):
            raise RuntimeError("The analyzer must be fitted before calling `summary()`.")
        columns = ["trend_slope", "trend_r2", "trend_pvalue", "significant_trend"]
        rows = [
            {self.id_col_: unique_id, **result}
            for unique_id, result in self.results_.items()
        ]
        return pd.DataFrame(rows, columns=[self.id_col_, *columns])
