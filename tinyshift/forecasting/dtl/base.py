# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import copy
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from tinyshift.series import detrend, hfi, hpi, select_pami_lag


class BaseDTL(BaseEstimator, RegressorMixin):
    """Shared LOWESS, trend-forecasting, and recombination workflow for DTL."""

    def __init__(
        self,
        residual_model_callable: Optional[Any] = None,
        freq: Optional[Union[str, int]] = None,
        trend_model_callable: Optional[Any] = None,
        trend_frac: float = 0.2,
        robust: bool = True,
        log_transform: bool = False,
        nlags: Any = "auto",
        pami_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.residual_model_callable = residual_model_callable
        self.freq = freq
        self.trend_model_callable = trend_model_callable
        self.trend_frac = trend_frac
        self.robust = robust
        self.log_transform = log_transform
        self.nlags = nlags
        self.pami_params = pami_params

    def _get_sku_config(self, config: Any, uid: Union[str, int]) -> Any:
        return config.get(uid) if isinstance(config, dict) else config

    def _get_model_cols(self, frame: pd.DataFrame) -> List[str]:
        return [
            column for column in frame if column not in (self.id_col_, self.time_col_)
        ]

    def _resolve_trend_factory(self, uid, default_factory):
        """Resolve the trend model factory configured for one SKU.

        The same ``default_factory`` reference is reused across every SKU that
        has no per-``unique_id`` override, which lets :meth:`fit` batch those
        series into a single panel-wide StatsForecast call.
        """
        factory = (
            self._get_sku_config(self.trend_model_callable, uid) or default_factory
        )
        if not callable(factory):
            raise TypeError("trend_model_callable must be callable.")
        return factory

    def _get_residual_lags(self, uid, residual_part: np.ndarray) -> List[int]:
        config = self._get_sku_config(self.nlags, uid)
        if config == "auto":
            selected, _, _ = select_pami_lag(residual_part, **(self.pami_params or {}))
            if isinstance(selected, int):
                return [selected] if selected > 0 else [1]
            return selected or [1]
        if isinstance(config, int) and not isinstance(config, bool):
            if config < 1:
                raise ValueError("nlags must be positive")
            return list(range(1, config + 1))
        if isinstance(config, list) and config:
            return config
        raise ValueError(f"Invalid lags configuration for unique_id {uid!r}: {config}")

    def _fit_panel(self, models, rows):
        """Fit one StatsForecast instance on a panel built from several series.

        Each row is ``(uid, values, dates)``. Every SKU still contributes its
        own values, but SKUs sharing the same resolved trend factory are
        concatenated into one panel so a single StatsForecast call fits all of
        them, instead of one call per SKU.
        """
        from statsforecast import StatsForecast

        frame = pd.concat(
            [
                pd.DataFrame(
                    {self.id_col_: uid, self.time_col_: dates, self.target_col_: values}
                )
                for uid, values, dates in rows
            ],
            ignore_index=True,
        )
        return StatsForecast(models=models, freq=self.freq_).fit(
            frame,
            id_col=self.id_col_,
            time_col=self.time_col_,
            target_col=self.target_col_,
        )

    def _fit_statsforecast(self, model, values, dates, uid):
        """Fit one StatsForecast instance for a single SKU.

        Kept for backward compatibility; :meth:`fit` uses :meth:`_fit_panel`
        directly so that SKUs sharing a trend factory can be batched together.
        """
        models = copy.deepcopy(model if isinstance(model, list) else [model])
        return self._fit_panel(models, [(uid, values, dates)])

    def _register_trend_row(
        self, trend_groups, uid_trend_key, trend_factory, uid, trend, dates
    ):
        """Assign one SKU's trend row to its shared-factory panel bucket."""
        trend_key = id(trend_factory)
        bucket = trend_groups.setdefault(
            trend_key, {"factory": trend_factory, "rows": []}
        )
        bucket["rows"].append((uid, trend, dates))
        uid_trend_key[uid] = trend_key

    def _fit_trend_panels(self, trend_groups):
        """Fit one StatsForecast panel per trend bucket."""
        return {
            key: self._fit_panel([bucket["factory"]()], bucket["rows"])
            for key, bucket in trend_groups.items()
        }

    def _make_residual_frame(self, group, residual_part):
        frame = group[[self.id_col_, self.time_col_] + self.exog_cols_].copy()
        frame[self.target_col_] = residual_part
        return frame

    def _fit_residuals(self, residuals, prediction_intervals, static_features):
        raise NotImplementedError

    def _predict_residuals(self, h, X_df, level):
        raise NotImplementedError

    def fit(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        prediction_intervals: Optional[Any] = None,
        static_features: Optional[List[str]] = None,
    ):
        from statsforecast.models import AutoETS

        if self.freq is None:
            raise ValueError(
                "Parameter 'freq' must be explicitly declared for DTLWrapper."
            )
        self.freq_ = self.freq
        self.id_col_, self.time_col_, self.target_col_ = id_col, time_col, target_col
        self.exog_cols_ = [c for c in df if c not in (id_col, time_col, target_col)]
        self.fitted_models_ = {}
        self.skus_nlags_ = {}

        processed = df[[id_col, time_col, target_col] + self.exog_cols_].copy()
        if self.log_transform:
            processed[target_col] = np.log1p(processed[target_col])
        decomposed = detrend(
            processed,
            frac=self.trend_frac,
            robust=self.robust,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )

        # Created once per fit() call so every SKU without a per-unique_id
        # override resolves to the SAME factory reference below, which lets
        # them be batched into one panel-wide StatsForecast call.
        default_trend_factory = partial(AutoETS, model="ZZN")

        trend_groups: Dict[int, Dict[str, Any]] = {}
        uid_trend_key: Dict[Union[str, int], int] = {}
        residuals = []
        for uid, group in df.groupby(id_col):
            group = group.sort_values(time_col).copy()
            trend_factory = self._resolve_trend_factory(uid, default_trend_factory)
            component = decomposed.loc[group.index]
            trend = component["trend"].to_numpy()
            residual = component["detrended"].to_numpy()
            lags = self._get_residual_lags(uid, residual)
            self.skus_nlags_[uid] = lags
            self._register_trend_row(
                trend_groups, uid_trend_key, trend_factory, uid, trend, group[time_col]
            )
            residuals.append((uid, self._make_residual_frame(group, residual), lags))

        trend_fitted = self._fit_trend_panels(trend_groups)
        for uid, key in uid_trend_key.items():
            self.fitted_models_[uid] = {"trend": trend_fitted[key]}

        self._fit_residuals(residuals, prediction_intervals, static_features)
        return self

    def _stabilize(self, values, method, weight):
        if method == "hpi":
            return hpi(values, w_s=weight)
        if method == "hfi":
            return hfi(values, w_s=weight)
        raise ValueError("stabilization_method must be 'hpi' or 'hfi'.")

    def _predict_component_for_uid(self, model, cache, uid, h):
        """Predict one shared trend model once and return one SKU's values."""
        if id(model) not in cache:
            cache[id(model)] = model.predict(h=h)
        frame = cache[id(model)]
        frame = frame[frame[self.id_col_] == uid]
        return frame[self._get_model_cols(frame)].sum(axis=1).to_numpy()

    def _recombine_uid_forecast(self, frame, trend, stabilization_method, w_s):
        """Add trend to each residual model column and post-process it."""
        for column in self._get_model_cols(frame):
            values = frame[column].to_numpy() + trend
            if self.log_transform:
                values = np.expm1(values)
            if stabilization_method is not None and w_s > 0:
                values = self._stabilize(values, stabilization_method, w_s)
            frame[column] = values
        return frame

    def predict(self, h, X_df=None, level=None, stabilization_method=None, w_s=0.0):
        if not hasattr(self, "fitted_models_") or not self.fitted_models_:
            raise RuntimeError(
                "The model must be fitted with .fit() before making predictions."
            )
        if self.exog_cols_ and X_df is None:
            raise ValueError(
                "X_df is required because the model was fitted with exogenous features."
            )
        residual_predictions = self._predict_residuals(h, X_df, level)
        # A trend model may be shared by several SKUs (panel fit), so each
        # distinct fitted object is predicted only once here.
        trend_cache: Dict[int, pd.DataFrame] = {}
        predictions = []
        for uid, models in self.fitted_models_.items():
            trend = self._predict_component_for_uid(
                models["trend"], trend_cache, uid, h
            )
            frame = residual_predictions[
                residual_predictions[self.id_col_] == uid
            ].copy()
            frame = self._recombine_uid_forecast(
                frame, trend, stabilization_method, w_s
            )
            predictions.append(frame)
        return pd.concat(predictions, ignore_index=True)
