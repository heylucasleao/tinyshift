"""Hierarchical distribution-parameter calibration for TSF."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .family import DistributionFamily


@dataclass
class Calibration:
    """Fitted dispersion layers and their fallback policy.

    ``between_group_variance`` stores the heterogeneity estimated for each
    shrunk layer.  Its values are variances of log-dispersion, rather than
    predictive variances of the calibrated distributions.
    """

    dispersion: dict[str, Any]
    between_group_variance: dict[str, float]

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
        """Estimate the dispersion hierarchy from out-of-fold forecasts.

        ``cv_df`` is expected to contain one observed target and one predicted
        conditional mean per cross-validation row.  The predictions should be
        out-of-fold: using in-sample predictions would make the residual
        variation too optimistic and, consequently, the predictive
        distributions too narrow.  Each row must also identify its series and
        forecast horizon through the columns configured in ``__init__``.

        The fitting flow is:

        1. Fit one **global** dispersion using every row.  Its logarithm is the
           root (parent) of the hierarchy.
        2. Fit a raw dispersion for each **horizon**, then shrink each log-fit
           toward the global log-dispersion.
        3. Fit a raw dispersion for each **series**, then shrink each log-fit
           toward the same global log-dispersion.
        4. Fit a raw dispersion for each **series x horizon** pair, then shrink
           it toward that series' already-shrunk log-dispersion.

        Shrinkage is performed on the log scale because dispersion parameters
        must remain positive.  For a group with raw estimate
        ``log_dispersion_raw``, estimation variance
        ``log_dispersion_estimation_variance``, parent ``parent``, and
        ``between_group_log_dispersion_variance``, the fitted value is::

            weight = between_group_log_dispersion_variance / (
                between_group_log_dispersion_variance
                + log_dispersion_estimation_variance
            )
            log_dispersion = (weight * log_dispersion_raw
                              + (1 - weight) * parent)
            dispersion = exp(log_dispersion)

        Thus, well-estimated groups retain more of their own value, while noisy
        or unidentifiable groups fall back toward their parent.  If the layer
        has fewer than two finite group fits, or its observed variation does
        not exceed the estimated fitting noise,
        ``between_group_log_dispersion_variance`` is zero and the whole layer
        collapses to its parents.

        Parameters
        ----------
        cv_df : pandas.DataFrame
            Out-of-fold calibration rows.  It must contain ``id_col``,
            ``target_col``, ``prediction_col``, and ``horizon_col``.  Target
            values must satisfy the support of ``family`` and predicted means
            must be finite.  Repeated rows for a group supply the observations
            used by that group's maximum-likelihood dispersion fit.

        Returns
        -------
        Calibration
            Immutable-by-convention fitted state containing:

            - ``dispersion['global']``: the unshrunk root estimate;
            - ``dispersion['global_horizon']``: horizon estimates shrunk toward
              the global estimate;
            - ``dispersion['series']``: series estimates shrunk toward the
              global estimate;
            - ``dispersion['series_horizon']``: pair estimates shrunk toward
              their corresponding series estimate;
            - ``between_group_variance``: the estimated log-dispersion
              variance between groups for each shrunk layer, useful for
              inspecting how heterogeneous that layer was.

            At prediction time, :meth:`Calibration.resolve` chooses the most
            specific available estimate.  Known series use
            series x horizon -> series -> global; unseen series use
            global x horizon -> global.

        Notes
        -----
        This method fits only the distribution's dispersion parameter.  The
        conditional means in ``prediction_col`` are treated as fixed outputs
        of the first-stage forecasting model; they are neither refitted nor
        altered here.  The input frame is read but not mutated.
        """
        global_dispersion, global_log_dispersion = self._fit_global(cv_df)
        global_horizon, global_horizon_between_group_variance = (
            self._fit_global_horizon(cv_df, global_log_dispersion)
        )
        series_fit, series, series_between_group_variance = self._fit_series(
            cv_df, global_log_dispersion
        )
        series_horizon, series_horizon_between_group_variance = (
            self._fit_series_horizon(cv_df, series_fit)
        )
        return Calibration(
            dispersion={
                "global": global_dispersion,
                "global_horizon": global_horizon,
                "series": series,
                "series_horizon": series_horizon,
            },
            between_group_variance={
                "global_horizon": global_horizon_between_group_variance,
                "series": series_between_group_variance,
                "series_horizon": series_horizon_between_group_variance,
            },
        )

    def _fit_global(self, cv_df: pd.DataFrame) -> tuple[float, float]:
        """Fit the global fallback and return it with its logarithm."""
        fitted, log_dispersion, _ = self.family.fit_log_dispersion(
            cv_df[self.target_col].to_numpy(),
            cv_df[self.prediction_col].to_numpy(),
        )
        return fitted, log_dispersion

    def _fit_global_horizon(
        self, cv_df: pd.DataFrame, global_log_dispersion: float
    ) -> tuple[dict[int, float], float]:
        """Shrink global-horizon dispersions toward the global fit."""
        fitted = self._fit_table(cv_df, [self.horizon_col])
        fitted["parent"] = global_log_dispersion
        fitted, between_group_log_dispersion_variance = self._shrink_layer(fitted)
        values = {
            int(row[self.horizon_col]): row["dispersion"]
            for _, row in fitted.iterrows()
        }
        return values, between_group_log_dispersion_variance

    def _fit_series(
        self, cv_df: pd.DataFrame, global_log_dispersion: float
    ) -> tuple[pd.DataFrame, dict[Any, float], float]:
        """Shrink per-series dispersions toward the global fit."""
        fitted = self._fit_table(cv_df, [self.id_col])
        fitted["parent"] = global_log_dispersion
        fitted, between_group_log_dispersion_variance = self._shrink_layer(fitted)
        values = dict(zip(fitted[self.id_col], fitted["dispersion"]))
        return fitted, values, between_group_log_dispersion_variance

    def _fit_series_horizon(
        self, cv_df: pd.DataFrame, series_fit: pd.DataFrame
    ) -> tuple[dict[tuple[Any, int], float], float]:
        """Shrink series×horizon dispersions toward their series fits."""
        fitted = self._fit_table(cv_df, [self.id_col, self.horizon_col])
        fitted["parent"] = fitted[self.id_col].map(
            series_fit.set_index(self.id_col)["log_dispersion"]
        )
        fitted, between_group_log_dispersion_variance = self._shrink_layer(fitted)
        values = {
            (row[self.id_col], int(row[self.horizon_col])): row["dispersion"]
            for _, row in fitted.iterrows()
        }
        return values, between_group_log_dispersion_variance

    def _fit_table(
        self, cv_df: pd.DataFrame, group_columns: list[str]
    ) -> pd.DataFrame:
        """Fit raw log-dispersion and its variance for calibration groups."""
        rows = []
        for keys, group in cv_df.groupby(group_columns, sort=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            _, log_dispersion, log_dispersion_estimation_variance = (
                self.family.fit_log_dispersion(
                    group[self.target_col].to_numpy(),
                    group[self.prediction_col].to_numpy(),
                )
            )
            rows.append(
                (*keys, log_dispersion, log_dispersion_estimation_variance)
            )
        return pd.DataFrame(
            rows,
            columns=[
                *group_columns,
                "log_dispersion_raw",
                "log_dispersion_estimation_variance",
            ],
        )

    def _shrink_layer(self, fitted: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Estimate between-group variance and shrink one fitted layer."""
        between_group_log_dispersion_variance = (
            self._between_group_log_dispersion_variance(fitted)
        )
        return (
            self._shrink(fitted, between_group_log_dispersion_variance),
            between_group_log_dispersion_variance,
        )

    @staticmethod
    def _between_group_log_dispersion_variance(fitted: pd.DataFrame) -> float:
        """Estimate log-dispersion variance between groups, net of fit noise."""
        finite = np.isfinite(fitted["log_dispersion_raw"]) & np.isfinite(
            fitted["log_dispersion_estimation_variance"]
        )
        if finite.sum() < 2:
            return 0.0
        residuals = (
            fitted.loc[finite, "log_dispersion_raw"] - fitted.loc[finite, "parent"]
        )
        observed = residuals.var(ddof=1)
        estimation_noise = fitted.loc[
            finite, "log_dispersion_estimation_variance"
        ].mean()
        return float(max(observed - estimation_noise, 0.0))

    @staticmethod
    def _shrink(
        fitted: pd.DataFrame, between_group_log_dispersion_variance: float
    ) -> pd.DataFrame:
        """Shrink log-dispersions toward their hierarchical parent."""
        fitted = fitted.copy()
        fitted["weight"] = np.where(
            np.isfinite(fitted["log_dispersion_estimation_variance"]),
            between_group_log_dispersion_variance
            / (
                between_group_log_dispersion_variance
                + fitted["log_dispersion_estimation_variance"]
            ),
            0.0,
        )
        fitted["log_dispersion"] = (
            fitted["weight"] * fitted["log_dispersion_raw"]
            + (1.0 - fitted["weight"]) * fitted["parent"]
        )
        fitted["dispersion"] = np.exp(fitted["log_dispersion"])
        return fitted
