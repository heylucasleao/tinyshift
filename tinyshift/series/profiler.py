from typing import Any, ClassVar

import numpy as np
import pandas as pd

from .diagnostic import trend_significance
from .entropy import theoretical_limit
from .analyzers import IntermittencyAnalyzer, SeasonalityAnalyzer
from .spectral import foreca, spectral_concentration


class SeriesProfiler:
    """Build a multi-dimensional diagnostic summary for panel time series.

    ``SeriesProfiler`` combines demand-occurrence, predictability, temporal,
    and spectral diagnostics. Input follows the Nixtla long-format convention:
    one row per observation, one series identifier column, one time column, and
    one numeric target column. Each series is ordered by time and analyzed
    independently.

    The target represents demand and must therefore contain finite,
    non-negative values.

    Parameters
    ----------
    adi_threshold : float, default=1.32
        ADI boundary between frequent and intermittent demand occurrence.
    cv2_threshold : float, default=0.49
        CV² boundary between low and high positive-demand variability. Together
        with ``adi_threshold``, it defines the demand class.
    top_k : int, default=2
        Maximum number of distinct candidate seasonal periods retained per
        series.
    noise_threshold_factor : float, default=2.0
        Multiplier applied to spectral background power when selecting
        significant seasonal peaks. Larger values make detection more
        conservative.
    fallback : int or list of int, optional
        Candidate periods used when no significant spectral peak is found.
    spectral_detrend : {"linear", "constant", "none"}, default="linear"
        Detrending strategy shared by ForeCA, spectral concentration, and
        seasonal-period detection.
    permutation_m : int, default=3
        Embedding dimension used to calculate permutation entropy and the
        theoretical predictability limit.
    permutation_delay : int, default=1
        Spacing between observations in each ordinal pattern.

    Attributes
    ----------
    results_ : dict
        Nested results indexed by unique ID and semantic section:
        ``demand_occurrence``, ``predictability``, ``temporal_structure``, and
        ``spectral_structure``.
    intermittency_ : IntermittencyAnalyzer
        Fitted analyzer containing the complete demand-occurrence results,
        including raw inter-demand intervals.
    seasonality_ : SeasonalityAnalyzer
        Fitted detector containing candidate periods and the underlying
        frequencies, power spectrum, and peak indices.
    id_col_, time_col_, target_col_ : str
        Panel column names recorded during :meth:`fit`.

    Metric definitions
    ------------------
    adi
        Average Demand Interval, ``N / N_positive``. A larger value indicates
        sparser demand. It is infinite when no positive demand occurs.
    cv2
        Squared coefficient of variation of strictly positive demand,
        ``(standard_deviation / mean)²``. A larger value indicates more variable
        non-zero demand sizes. It is undefined when demand is always zero.
    zero_prop
        Fraction of observations equal to zero, bounded by 0 and 1.
    interval_cv
        Coefficient of variation of distances between consecutive positive
        demands. A larger value means occurrence timing is less regular. It is
        undefined when fewer than two inter-demand intervals are available.
    class
        ADI-CV² demand classification: ``smooth``, ``intermittent``,
        ``erratic``, or ``lumpy``. It is ``None`` when classification is
        undefined.
    foreca
        Forecastability based on normalized spectral entropy, bounded by 0 and
        1. Higher values indicate a more concentrated and structured spectrum.
    limit
        Ordinal predictability limit, calculated as one minus normalized
        permutation entropy. Values closer to 1 indicate more regular ordinal
        patterns.
    trend_r2
        R² from a linear regression of the target against observation order.
        Higher values indicate that a linear trend explains more variance.
    trend_pvalue
        P-value for the null hypothesis that the linear trend slope is zero.
        Smaller values provide stronger evidence of a non-zero linear trend.
    spectral_conc
        Normalized spectral concentration, bounded by 0 and 1. Values closer to
        1 indicate that power is concentrated in fewer frequency components.
    candidate_periods
        Significant FFT-derived periods expressed in numbers of observations.
        Periods longer than half the series length are excluded. An empty list
        means no candidate was found and no fallback was configured.

    Examples
    --------
    >>> profiler = SeriesProfiler(top_k=2)
    >>> profiler.fit(df, id_col="unique_id", time_col="ds", target_col="y")
    SeriesProfiler(...)
    >>> profiler.summary()
      unique_id  adi  cv2  ...  spectral_conc  candidate_periods
    0         A  ...  ...  ...            ...             [7, 28]

    Inspect one semantic section without flattening the results:

    >>> profiler.results_["A"]["predictability"]
    {'foreca': ..., 'limit': ...}
    """

    summary_columns: ClassVar[tuple[str, ...]] = (
        "adi",
        "cv2",
        "zero_prop",
        "interval_cv",
        "class",
        "foreca",
        "limit",
        "trend_r2",
        "trend_pvalue",
        "spectral_conc",
        "candidate_periods",
    )

    def __init__(
        self,
        adi_threshold: float = 1.32,
        cv2_threshold: float = 0.49,
        top_k: int = 2,
        noise_threshold_factor: float = 2.0,
        fallback: int | list[int] | None = None,
        spectral_detrend: str = "linear",
        permutation_m: int = 3,
        permutation_delay: int = 1,
    ) -> None:
        self.adi_threshold = adi_threshold
        self.cv2_threshold = cv2_threshold
        self.top_k = top_k
        self.noise_threshold_factor = noise_threshold_factor
        self.fallback = fallback
        self.spectral_detrend = spectral_detrend
        self.permutation_m = permutation_m
        self.permutation_delay = permutation_delay

    def __repr__(self) -> str:
        return (
            "SeriesProfiler("
            f"adi_threshold={self.adi_threshold}, "
            f"cv2_threshold={self.cv2_threshold}, "
            f"top_k={self.top_k}, "
            f"noise_threshold_factor={self.noise_threshold_factor}, "
            f"fallback={self.fallback!r}, "
            f"spectral_detrend={self.spectral_detrend!r}, "
            f"permutation_m={self.permutation_m}, "
            f"permutation_delay={self.permutation_delay}"
            ")"
        )

    @staticmethod
    def demand_occurrence(result: dict[str, Any]) -> dict[str, Any]:
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
    ) -> dict[str, float]:
        """Calculate predictability diagnostics for one series."""
        return {
            "foreca": foreca(values, detrend=detrend),
            "limit": theoretical_limit(
                values, m=permutation_m, delay=permutation_delay
            ),
        }

    @staticmethod
    def temporal_structure(
        values: np.ndarray,
    ) -> dict[str, float]:
        """Calculate temporal-structure diagnostics for one series."""
        trend_r2, trend_pvalue = trend_significance(values)
        return {
            "trend_r2": trend_r2,
            "trend_pvalue": trend_pvalue,
        }

    @staticmethod
    def spectral_structure(
        values: np.ndarray,
        result: dict[str, Any],
        *,
        detrend: str,
    ) -> dict[str, Any]:
        """Calculate and select spectral-structure diagnostics."""
        return {
            "spectral_conc": spectral_concentration(values, detrend=detrend),
            "candidate_periods": result["candidate_periods"],
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

    def fit(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> "SeriesProfiler":
        """Calculate every diagnostic section for each panel series.

        Parameters
        ----------
        df : pandas.DataFrame
            Nixtla-style panel containing the ID, time, and target columns.
        id_col : str, default="unique_id"
            Column identifying independent series.
        time_col : str, default="ds"
            Column used to order observations within each series.
        target_col : str, default="y"
            Numeric column containing finite, non-negative demand values.

        Returns
        -------
        SeriesProfiler
            Fitted profiler instance.
        """
        self._validate_panel(df, id_col, time_col, target_col)
        self.id_col_ = id_col
        self.time_col_ = time_col
        self.target_col_ = target_col

        self.intermittency_ = IntermittencyAnalyzer(
            adi_threshold=self.adi_threshold,
            cv2_threshold=self.cv2_threshold,
        ).fit(df, id_col=id_col, time_col=time_col, target_col=target_col)
        self.seasonality_ = SeasonalityAnalyzer(
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
                    values,
                ),
                "spectral_structure": self.spectral_structure(
                    values,
                    self.seasonality_.results_[unique_id],
                    detrend=self.spectral_detrend,
                ),
            }
        return self

    def summary(self) -> pd.DataFrame:
        """Return the fitted diagnostics as one flat row per series.

        Returns
        -------
        pandas.DataFrame
            Columns are the fitted ID column followed by ``adi``, ``cv2``,
            ``zero_prop``, ``interval_cv``, ``class``, ``foreca``, ``limit``,
            ``trend_r2``, ``trend_pvalue``, ``spectral_conc``, and
            ``candidate_periods``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not hasattr(self, "results_"):
            raise RuntimeError(
                "The profiler must be fitted before calling `summary()`."
            )

        rows = []
        for unique_id, sections in self.results_.items():
            row = {self.id_col_: unique_id}
            for section in sections.values():
                row.update(section)
            rows.append(row)
        return pd.DataFrame(rows, columns=[self.id_col_, *self.summary_columns])
