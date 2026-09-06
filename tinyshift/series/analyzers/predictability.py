"""Complementary predictability diagnostics for panel time series."""

import numpy as np
import pandas as pd

from ..entropy import theoretical_limit
from ..spectral import foreca, spectral_concentration
from .base import BaseSeriesAnalyzer


class PredictabilityAnalyzer(BaseSeriesAnalyzer):
    """Measure spectral and ordinal structure for each panel series."""

    def __init__(
        self,
        detrend: str = "linear",
        permutation_m: int = 3,
        permutation_delay: int = 1,
    ) -> None:
        self.detrend = detrend
        self.permutation_m = permutation_m
        self.permutation_delay = permutation_delay
        if detrend not in {"linear", "constant", "none"}:
            raise ValueError("'detrend' must be one of {'linear', 'constant', 'none'}.")
        if isinstance(permutation_m, bool) or not isinstance(permutation_m, int) or permutation_m < 2:
            raise ValueError("'permutation_m' must be an integer greater than or equal to 2.")
        if isinstance(permutation_delay, bool) or not isinstance(permutation_delay, int) or permutation_delay < 1:
            raise ValueError("'permutation_delay' must be a positive integer.")

    def __repr__(self) -> str:
        return (
            "PredictabilityAnalyzer("
            f"detrend={self.detrend!r}, "
            f"permutation_m={self.permutation_m}, "
            f"permutation_delay={self.permutation_delay}"
            ")"
        )

    def _validate_target(self, df: pd.DataFrame, target_col: str) -> None:
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise ValueError(f"Target column {target_col!r} must be numeric.")
        if not np.isfinite(df[target_col].to_numpy(dtype=float)).all():
            raise ValueError("Target values must be finite.")

    def _fit_single(self, values: pd.Series) -> dict[str, float]:
        return {
            "foreca": foreca(values, detrend=self.detrend),
            "limit": theoretical_limit(
                values, m=self.permutation_m, delay=self.permutation_delay
            ),
            "spectral_concentration": spectral_concentration(
                values, detrend=self.detrend
            ),
        }

    def summary(self) -> pd.DataFrame:
        """Return predictability diagnostics with one row per series."""
        if not hasattr(self, "results_"):
            raise RuntimeError("The analyzer must be fitted before calling `summary()`.")
        columns = ["foreca", "limit", "spectral_concentration"]
        rows = [
            {self.id_col_: unique_id, **result}
            for unique_id, result in self.results_.items()
        ]
        return pd.DataFrame(rows, columns=[self.id_col_, *columns])
