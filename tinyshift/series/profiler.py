from collections import Counter
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .diagnostic import hurst_exponent, trend_significance
from .forecastability import foreca, spectral_concentration, theoretical_limit
from .intermittency import IntermittencyAnalyzer
from .seasonality import SeasonalPeriodDetector


class SeriesProfiler:
    """Build a structured diagnostic profile for a Nixtla-style panel."""

    profile_columns = [
        "adi",
        "cv2",
        "zero_prop",
        "interval_cv",
        "class",
        "foreca",
        "limit",
        "hurst",
        "trend_r2",
        "trend_pvalue",
        "spectral_conc",
        "periods",
    ]

    def __init__(
        self,
        adi_threshold: float = 1.32,
        cv2_threshold: float = 0.49,
        top_k: int = 2,
        noise_threshold_factor: float = 2.0,
        fallback: Optional[Union[int, List[int]]] = None,
        spectral_detrend: str = "linear",
        hurst_d: int = 1,
        permutation_m: int = 3,
        permutation_delay: int = 1,
    ) -> None:
        self.adi_threshold = adi_threshold
        self.cv2_threshold = cv2_threshold
        self.top_k = top_k
        self.noise_threshold_factor = noise_threshold_factor
        self.fallback = fallback
        self.spectral_detrend = spectral_detrend
        self.hurst_d = hurst_d
        self.permutation_m = permutation_m
        self.permutation_delay = permutation_delay

    @staticmethod
    def demand_occurrence(result: Dict[str, Any]) -> Dict[str, Any]:
        """Select and name demand-occurrence diagnostics."""
        return {
            "adi": result["adi"],
            "cv2": result["cv2"],
            "zero_prop": result["zero_proportion"],
            "interval_cv": result["interval_cv"],
            "class": result["classification"],
        }

    @staticmethod
    def predictability(
        values: np.ndarray,
        *,
        detrend: str,
        permutation_m: int,
        permutation_delay: int,
    ) -> Dict[str, float]:
        """Calculate predictability diagnostics for one series."""
        return {
            "foreca": foreca(values, detrend=detrend),
            "limit": theoretical_limit(
                values, m=permutation_m, delay=permutation_delay
            ),
        }

    @staticmethod
    def temporal_structure(values: np.ndarray, *, hurst_d: int) -> Dict[str, float]:
        """Calculate temporal-structure diagnostics for one series."""
        hurst, _ = hurst_exponent(values, d=hurst_d)
        trend_r2, trend_pvalue = trend_significance(values)
        return {
            "hurst": hurst,
            "trend_r2": trend_r2,
            "trend_pvalue": trend_pvalue,
        }

    @staticmethod
    def spectral_structure(
        values: np.ndarray,
        result: Dict[str, Any],
        *,
        detrend: str,
    ) -> Dict[str, Any]:
        """Calculate and select spectral-structure diagnostics."""
        return {
            "spectral_conc": spectral_concentration(values, detrend=detrend),
            "periods": result["periods"],
        }

    @staticmethod
    def _validate_panel(
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
    ) -> None:
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

        counts = df.groupby(id_col, observed=True).size()
        short_ids = counts[counts < 30].index.tolist()
        if short_ids:
            raise ValueError(
                "Each series must contain at least 30 observations for the Hurst "
                f"exponent; short series: {short_ids}."
            )

    def fit(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> "SeriesProfiler":
        """Calculate every profile section for each series in the panel."""
        self._validate_panel(df, id_col, time_col, target_col)
        self.id_col_ = id_col
        self.time_col_ = time_col
        self.target_col_ = target_col

        self.intermittency_ = IntermittencyAnalyzer(
            adi_threshold=self.adi_threshold,
            cv2_threshold=self.cv2_threshold,
        ).fit(df, id_col=id_col, time_col=time_col, target_col=target_col)
        self.seasonality_ = SeasonalPeriodDetector(
            top_k=self.top_k,
            noise_threshold_factor=self.noise_threshold_factor,
            fallback=self.fallback,
            detrend=self.spectral_detrend,
        ).fit(df, id_col=id_col, time_col=time_col, target_col=target_col)

        self.results_ = {}
        ordered = df.sort_values([id_col, time_col])
        for unique_id, group in ordered.groupby(id_col, sort=False, observed=True):
            values = group[target_col].to_numpy(dtype=float)
            self.results_[unique_id] = {
                "demand_occurrence": self.demand_occurrence(
                    self.intermittency_.results_[unique_id]
                ),
                "predictability": self.predictability(
                    values,
                    detrend=self.spectral_detrend,
                    permutation_m=self.permutation_m,
                    permutation_delay=self.permutation_delay,
                ),
                "temporal_structure": self.temporal_structure(
                    values, hurst_d=self.hurst_d
                ),
                "spectral_structure": self.spectral_structure(
                    values,
                    self.seasonality_.results_[unique_id],
                    detrend=self.spectral_detrend,
                ),
            }
        return self

    def profile(self) -> pd.DataFrame:
        """Return the nested results as one flat row per series."""
        if not hasattr(self, "results_"):
            raise RuntimeError(
                "The profiler must be fitted before calling `profile()`."
            )

        rows = []
        for unique_id, sections in self.results_.items():
            row = {self.id_col_: unique_id}
            for section in sections.values():
                row.update(section)
            rows.append(row)
        return pd.DataFrame(rows, columns=[self.id_col_, *self.profile_columns])

    def summary(self) -> Dict[str, Any]:
        """Return aggregate numeric, demand-class, and seasonal summaries."""
        profile = self.profile()
        numeric = profile.drop(columns=self.id_col_).select_dtypes(include=[np.number])
        metrics = numeric.describe(percentiles=[0.5]).T.rename(
            columns={"50%": "median"}
        )
        class_counts = profile["class"].value_counts(dropna=False)
        period_counts = Counter(
            period for periods in profile["periods"] for period in periods
        )
        return {
            "n_series": len(profile),
            "metrics": metrics,
            "demand_classes": class_counts.to_dict(),
            "period_frequency": dict(sorted(period_counts.items())),
        }
