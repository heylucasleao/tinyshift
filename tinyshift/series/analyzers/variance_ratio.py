# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from numbers import Real
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from ..diagnostic import variance_ratio
from .base import BaseSeriesAnalyzer


class VarianceRatioAnalyzer(BaseSeriesAnalyzer):
    """Diagnose serial dependence across multiple aggregation horizons.

    The analyzer evaluates the variance-ratio statistic for each series in a
    panel across one or more horizons. Values near one indicate weak serial
    dependence, whereas systematic deviations from one suggest persistence or
    mean-reversion at the evaluated scale.

    Parameters
    ----------
    horizons : list of int, optional
        Aggregation horizons to evaluate. When omitted, a power-of-two grid is
        constructed automatically for each series up to the smaller of one tenth
        of the series length and ``max_horizon``.
    max_horizon : int, default=64
        Maximum horizon considered when generating default values automatically.
    significance_level : float, default=0.05
        P-value threshold used for ``significant_dependence``. Must lie
        strictly between zero and one.

    Attributes
    ----------
    results_ : dict
        Mapping from each unique ID to a nested dictionary keyed by horizon. Each
        horizon entry contains ``variance_ratio``, ``z_statistic``, ``p_value``,
        and ``significant_dependence``.

    Notes
    -----
    The test is computed on the original series and aggregated increments. The
    variance-ratio statistic compares the variance of the aggregated process with
    the variance implied by independent one-step shocks. The implementation is
    most informative when the fitted panel is sufficiently long to support the
    requested horizons.

    Examples
    --------
    >>> analyzer = VarianceRatioAnalyzer(horizons=[2, 4, 8])
    >>> analyzer.fit(df, id_col="unique_id", time_col="ds", target_col="y")
    VarianceRatioAnalyzer(...)
    >>> analyzer.summary().head()
            unique_id  horizon  variance_ratio  significant_dependence
        0        A        2            ...                    ...
    """

    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        max_horizon: int = 64,
        significance_level: float = 0.05,
    ) -> None:
        self.horizons = horizons
        self.max_horizon = max_horizon
        self.significance_level = significance_level
        self._validate_params()

    def __repr__(self) -> str:
        return (
            "VarianceRatioAnalyzer("
            f"horizons={self.horizons!r}, "
            f"max_horizon={self.max_horizon}, "
            f"significance_level={self.significance_level}"
            ")"
        )

    def _validate_params(self) -> None:
        if (
            isinstance(self.max_horizon, bool)
            or not isinstance(self.max_horizon, int)
            or self.max_horizon <= 1
        ):
            raise ValueError("'max_horizon' must be an integer greater than 1.")

        if (
            isinstance(self.significance_level, bool)
            or not isinstance(self.significance_level, Real)
            or not np.isfinite(self.significance_level)
            or not 0 < self.significance_level < 1
        ):
            raise ValueError("'significance_level' must be between 0 and 1.")

        if self.horizons is None:
            return

        if not isinstance(self.horizons, (list, tuple)) or not self.horizons:
            raise ValueError("'horizons' must be a non-empty list of integers.")
        if any(
            isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 1
            for horizon in self.horizons
        ):
            raise ValueError("Every horizon must be an integer greater than 1.")

    def _default_horizons(self, n: int) -> List[int]:
        """Generate power-of-two horizons supported by the sample size."""
        max_horizon = min(n // 10, self.max_horizon)
        return [1 << exponent for exponent in range(1, max_horizon.bit_length())]

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
                "significant_dependence": bool(
                    np.isfinite(p_value) and p_value < self.significance_level
                ),
            }

        return results

    def _validate_target(self, df: pd.DataFrame, target_col: str) -> None:
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise ValueError(f"Target column {target_col!r} must be numeric.")
        if not np.isfinite(df[target_col].to_numpy(dtype=float)).all():
            raise ValueError("Target values must be finite.")

    def summary(self) -> pd.DataFrame:
        """Return one row per series and evaluated horizon.

        Returns
        -------
        pandas.DataFrame
            Long-format table containing one row per ``(unique_id, horizon)``
            pair, the variance-ratio estimate, and its significance flag. The
            z-statistic and p-value remain available in ``results_``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not hasattr(self, "results_"):
            raise RuntimeError(
                "The analyzer must be fitted before calling `summary()`."
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
            "significant_dependence",
        ]
        return pd.DataFrame(rows, columns=columns)
