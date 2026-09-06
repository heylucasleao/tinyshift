# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .dependence import permutation_auto_mutual_information

LagMode = Literal["short", "point", "range"]


@dataclass(frozen=True)
class PAMIResult:
    """PAMI curve and the lag values at all of its local minima."""

    taus: np.ndarray
    values: np.ndarray
    local_minima: list[int]


def _format_lags(
    local_minima: Sequence[int],
    *,
    mode: LagMode,
    fallback: int,
    short: int,
) -> list[int]:
    """Create model lags from the first PAMI minimum or a fallback."""
    if isinstance(fallback, bool) or not isinstance(fallback, int) or fallback < 1:
        raise ValueError("'fallback' must be a positive integer.")
    if isinstance(short, bool) or not isinstance(short, int) or short < 1:
        raise ValueError("'short' must be a positive integer.")

    selected = int(local_minima[0]) if local_minima else fallback
    if mode == "point":
        return [selected]
    if mode == "short":
        return sorted({*range(1, min(short, selected) + 1), selected})
    if mode == "range":
        return list(range(1, selected + 1))
    raise ValueError("'mode' must be one of {'short', 'point', 'range'}.")


def create_pami_lags(
    local_minima: Mapping[Any, Sequence[int]],
    *,
    mode: LagMode = "range",
    fallback: int = 1,
    short: int = 1,
) -> dict[Any, list[int]]:
    """Create a DTL/DMSTL-compatible lag dictionary from PAMI minima."""
    return {
        unique_id: _format_lags(
            minima,
            mode=mode,
            fallback=fallback,
            short=short,
        )
        for unique_id, minima in local_minima.items()
    }


class PAMIAnalyzer:
    """Find local minima in PAMI curves for one or more time series."""

    def __init__(
        self,
        max_tau: int = 365,
        m: int = 3,
        delay: int = 1,
        normalize: bool = False,
    ) -> None:
        self.max_tau = max_tau
        self.m = m
        self.delay = delay
        self.normalize = normalize
        self._validate_params()

    def __repr__(self) -> str:
        return (
            "PAMIAnalyzer("
            f"max_tau={self.max_tau}, m={self.m}, delay={self.delay}, "
            f"normalize={self.normalize}"
            ")"
        )

    def _validate_params(self) -> None:
        for name, value, minimum in (
            ("max_tau", self.max_tau, 1),
            ("m", self.m, 2),
            ("delay", self.delay, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"'{name}' must be an integer greater than or equal to {minimum}.")
        if not isinstance(self.normalize, bool):
            raise TypeError("'normalize' must be a boolean.")

    def analyze(self, values: Sequence[float]) -> PAMIResult:
        """Calculate a PAMI curve and locate all of its local minima."""
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("Input data must be 1-dimensional.")
        if not np.isfinite(values).all():
            raise ValueError("Input data must contain only finite values.")

        max_valid_tau = len(values) - (self.m - 1) * self.delay - 1
        max_tau = min(self.max_tau, max_valid_tau)
        if max_tau < 1:
            raise ValueError("Time series is too short for the given m and delay.")

        taus = np.arange(1, max_tau + 1)
        pami_values = np.asarray(
            [
                permutation_auto_mutual_information(
                    values,
                    tau=int(tau),
                    m=self.m,
                    delay=self.delay,
                    normalize=self.normalize,
                )
                for tau in taus
            ],
            dtype=np.float64,
        )
        positions, _ = find_peaks(-pami_values)
        minima = taus[positions].astype(int).tolist()
        return PAMIResult(taus=taus, values=pami_values, local_minima=minima)

    def fit(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> "PAMIAnalyzer":
        """Analyze every time series in a panel DataFrame."""
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

        self.id_col_ = id_col
        self.time_col_ = time_col
        self.target_col_ = target_col
        ordered = df.sort_values([id_col, time_col])
        self.results_ = {
            unique_id: self.analyze(group[target_col].to_numpy(dtype=float))
            for unique_id, group in ordered.groupby(id_col, sort=False, observed=True)
        }
        return self

    def summary(self) -> pd.DataFrame:
        """Return only the ID and local minima found for each series."""
        if not hasattr(self, "results_"):
            raise RuntimeError("The analyzer must be fitted before calling `summary()`.")
        return pd.DataFrame(
            [
                {self.id_col_: unique_id, "local_minima": result.local_minima}
                for unique_id, result in self.results_.items()
            ],
            columns=[self.id_col_, "local_minima"],
        )

    def lags(
        self,
        *,
        mode: LagMode = "range",
        fallback: int = 1,
        short: int = 1,
    ) -> dict[Any, list[int]]:
        """Create a model-compatible lag dictionary from fitted minima."""
        if not hasattr(self, "results_"):
            raise RuntimeError("The analyzer must be fitted before creating lags.")
        return create_pami_lags(
            {uid: result.local_minima for uid, result in self.results_.items()},
            mode=mode,
            fallback=fallback,
            short=short,
        )
