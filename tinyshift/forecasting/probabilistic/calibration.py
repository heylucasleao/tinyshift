"""Hierarchical distribution-parameter calibration for TSF."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .family import DistributionFamily


@dataclass
class Calibration:
    """Fitted dispersion layers and their fallback policy."""

    dispersion: dict[str, Any]
    tau2: dict[str, float]

    def resolve(self, uid: Any, horizon: int) -> float:
        """Resolve one dispersion through the fitted fallback hierarchy.

        Resolution distinguishes known series from cold starts. A known
        ``(series, horizon)`` uses the most specific shrunk estimate. When that
        horizon was not calibrated, the series-level estimate is preferred
        because it retains information learned for that series. For an unknown
        series, no series-level evidence exists, so the shrunk global-horizon
        estimate is used. The global estimate is the final fallback when
        neither dimension has a calibrated value.

        The resulting precedence is::

            known series:   series×horizon -> series -> global
            unknown series: global×horizon -> global

        All non-global layers have already been regularized toward their
        statistical parent during fitting; this method only selects a fitted
        value and does not perform additional shrinkage.
        """
        series_horizon = self.dispersion["series_horizon"]
        if (uid, horizon) in series_horizon:
            return float(series_horizon[uid, horizon])

        series = self.dispersion["series"]
        if uid in series:
            return float(series[uid])

        global_horizon = self.dispersion["global_horizon"]
        if horizon in global_horizon:
            return float(global_horizon[horizon])

        return float(self.dispersion["global"])


class Calibrator:
    """Estimate hierarchical dispersions from out-of-fold predictions."""

    def __init__(
        self,
        family: DistributionFamily,
        id_col: str,
        target_col: str,
        prediction_col: str,
        horizon_col: str = "_horizon",
    ) -> None:
        self.family = family
        self.id_col = id_col
        self.target_col = target_col
        self.prediction_col = prediction_col
        self.horizon_col = horizon_col

    def fit(self, cv_df: pd.DataFrame) -> Calibration:
        """Fit and shrink every layer of the dispersion hierarchy."""
        global_dispersion, global_theta = self._fit_global(cv_df)
        global_horizon, tau2_global_horizon = self._fit_global_horizon(
            cv_df, global_theta
        )
        series_fit, series, tau2_series = self._fit_series(cv_df, global_theta)
        series_horizon, tau2_series_horizon = self._fit_series_horizon(
            cv_df, series_fit
        )
        return Calibration(
            dispersion={
                "global": global_dispersion,
                "global_horizon": global_horizon,
                "series": series,
                "series_horizon": series_horizon,
            },
            tau2={
                "global_horizon": tau2_global_horizon,
                "series": tau2_series,
                "series_horizon": tau2_series_horizon,
            },
        )

    def _fit_global(self, cv_df: pd.DataFrame) -> tuple[float, float]:
        """Fit the global fallback and return it with its logarithm."""
        fitted, theta, _ = self.family.fit_log_dispersion(
            cv_df[self.target_col].to_numpy(),
            cv_df[self.prediction_col].to_numpy(),
        )
        return fitted, theta

    def _fit_global_horizon(
        self, cv_df: pd.DataFrame, global_theta: float
    ) -> tuple[dict[int, float], float]:
        """Shrink global-horizon dispersions toward the global fit."""
        fitted = self._fit_table(cv_df, [self.horizon_col])
        fitted["parent"] = global_theta
        fitted, tau2 = self._shrink_layer(fitted)
        values = {
            int(row[self.horizon_col]): row["dispersion"]
            for _, row in fitted.iterrows()
        }
        return values, tau2

    def _fit_series(
        self, cv_df: pd.DataFrame, global_theta: float
    ) -> tuple[pd.DataFrame, dict[Any, float], float]:
        """Shrink per-series dispersions toward the global fit."""
        fitted = self._fit_table(cv_df, [self.id_col])
        fitted["parent"] = global_theta
        fitted, tau2 = self._shrink_layer(fitted)
        values = dict(zip(fitted[self.id_col], fitted["dispersion"]))
        return fitted, values, tau2

    def _fit_series_horizon(
        self, cv_df: pd.DataFrame, series_fit: pd.DataFrame
    ) -> tuple[dict[tuple[Any, int], float], float]:
        """Shrink series×horizon dispersions toward their series fits."""
        fitted = self._fit_table(cv_df, [self.id_col, self.horizon_col])
        fitted["parent"] = fitted[self.id_col].map(
            series_fit.set_index(self.id_col)["theta"]
        )
        fitted, tau2 = self._shrink_layer(fitted)
        values = {
            (row[self.id_col], int(row[self.horizon_col])): row["dispersion"]
            for _, row in fitted.iterrows()
        }
        return values, tau2

    def _fit_table(
        self, cv_df: pd.DataFrame, group_columns: list[str]
    ) -> pd.DataFrame:
        """Fit raw log-dispersion and its variance for calibration groups."""
        rows = []
        for keys, group in cv_df.groupby(group_columns, sort=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            _, theta, variance = self.family.fit_log_dispersion(
                group[self.target_col].to_numpy(),
                group[self.prediction_col].to_numpy(),
            )
            rows.append((*keys, theta, variance))
        return pd.DataFrame(rows, columns=[*group_columns, "theta_raw", "variance"])

    def _shrink_layer(self, fitted: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Estimate between-group variance and shrink one fitted layer."""
        tau2 = self._between_group_variance(fitted)
        return self._shrink(fitted, tau2), tau2

    @staticmethod
    def _between_group_variance(fitted: pd.DataFrame) -> float:
        """Estimate genuine between-group variance after removing fit noise."""
        finite = np.isfinite(fitted["theta_raw"]) & np.isfinite(fitted["variance"])
        if finite.sum() < 2:
            return 0.0
        residuals = fitted.loc[finite, "theta_raw"] - fitted.loc[finite, "parent"]
        observed = residuals.var(ddof=1)
        estimation_noise = fitted.loc[finite, "variance"].mean()
        return float(max(observed - estimation_noise, 0.0))

    @staticmethod
    def _shrink(fitted: pd.DataFrame, tau2: float) -> pd.DataFrame:
        """Shrink log-dispersions toward their hierarchical parent."""
        fitted = fitted.copy()
        fitted["weight"] = np.where(
            np.isfinite(fitted["variance"]),
            tau2 / (tau2 + fitted["variance"]),
            0.0,
        )
        fitted["theta"] = (
            fitted["weight"] * fitted["theta_raw"]
            + (1.0 - fitted["weight"]) * fitted["parent"]
        )
        fitted["dispersion"] = np.exp(fitted["theta"])
        return fitted
