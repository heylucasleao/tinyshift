# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ..dependence import permutation_auto_mutual_information
from .base import BaseSeriesAnalyzer

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


class PAMIAnalyzer(BaseSeriesAnalyzer):
    """Detect candidate lags from permutation auto-mutual information curves.

    The analyzer computes a permutation auto-mutual information (PAMI) curve for
    each series in a panel and records the local minima of the resulting curve.
    Those minima are interpreted as candidate lag values associated with the
    strongest non-linear dependence structure in the signal.

    Parameters
    ----------
    max_tau : int, default=365
        Maximum lag to evaluate. The effective horizon is truncated by the
        available sample size for the chosen embedding order and delay.
    m : int, default=3
        Embedding dimension used by the permutation auto-mutual information
        estimator.
    delay : int, default=1
        Spacing between observations used to form each ordinal pattern.
    normalize : bool, default=False
        Whether to normalize the permutation auto-mutual information values.

    Attributes
    ----------
    results_ : dict
        Mapping from each unique ID to a :class:`PAMIResult` object containing the
        evaluated lags, the PAMI curve, and the local minima found in that curve.

    Notes
    -----
    PAMI measures dependence between a series and its lagged version using
    ordinal-pattern entropy. Local minima often correspond to informative lags
    where the memory structure is strongest or most rhythmically structured.

    The class follows the panel-oriented workflow of
    :class:`~tinyshift.series.analyzers.base.BaseSeriesAnalyzer`: every ID is
    independently fitted, sorted by time, and summarized through a compact
    per-series result table.

    Examples
    --------
    >>> analyzer = PAMIAnalyzer(max_tau=60, m=3, delay=1)
    >>> analyzer.fit(df, id_col="unique_id", time_col="ds", target_col="y")
    PAMIAnalyzer(...)
    >>> analyzer.summary().head()
      unique_id                 local_minima
    0       A                        [7, 14]
    >>> analyzer.lags(mode="short", fallback=1, short=3)
    {'A': [1, 2, 7, 14]}

    See Also
    --------
    create_pami_lags : Construct a model-compatible lag dictionary from minima.
    """

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
                raise ValueError(
                    f"'{name}' must be an integer greater than or equal to {minimum}."
                )
        if not isinstance(self.normalize, bool):
            raise TypeError("'normalize' must be a boolean.")

    def analyze(self, values: Sequence[float]) -> PAMIResult:
        """Calculate a PAMI curve and locate all of its local minima.

        Parameters
        ----------
        values : sequence of float
            Univariate time series used to compute the PAMI curve.

        Returns
        -------
        PAMIResult
            Object containing the evaluated lag values, the corresponding PAMI
            curve, and the local minima of that curve.

        Raises
        ------
        ValueError
            If the input is not one-dimensional, contains non-finite values, or
            is too short for the configured embedding dimension and delay.
        """
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

    def _validate_target(self, df: pd.DataFrame, target_col: str) -> None:
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise ValueError(f"Target column {target_col!r} must be numeric.")
        if not np.isfinite(df[target_col].to_numpy(dtype=float)).all():
            raise ValueError("Target values must be finite.")

    def _fit_single(self, values: pd.Series) -> PAMIResult:
        return self.analyze(values)

    def summary(self) -> pd.DataFrame:
        """Return the detected lag minima for every fitted series.

        Returns
        -------
        pandas.DataFrame
            A two-column table with the series identifier and the local minima
            discovered in its PAMI curve.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not hasattr(self, "results_"):
            raise RuntimeError(
                "The analyzer must be fitted before calling `summary()`."
            )
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
        """Create a model-compatible lag dictionary from fitted minima.

        Parameters
        ----------
        mode : {"short", "point", "range"}, default="range"
            Strategy used to build the final lag list from each series minima.
        fallback : int, default=1
            Integer used when no valid local minimum is available.
        short : int, default=1
            Number of short lags to include when ``mode="short"``.

        Returns
        -------
        dict
            Mapping from each unique ID to the selected lag values.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not hasattr(self, "results_"):
            raise RuntimeError("The analyzer must be fitted before creating lags.")
        return create_pami_lags(
            {uid: result.local_minima for uid, result in self.results_.items()},
            mode=mode,
            fallback=fallback,
            short=short,
        )
