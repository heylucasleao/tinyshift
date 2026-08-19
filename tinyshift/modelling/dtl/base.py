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
        self.pami_params = pami_params or {}

    def _get_sku_config(self, config: Any, uid: Union[str, int]) -> Any:
        return config.get(uid) if isinstance(config, dict) else config

    def _get_model_cols(self, frame: pd.DataFrame) -> List[str]:
        return [
            column for column in frame if column not in (self.id_col_, self.time_col_)
        ]

    def _get_trend_config(self, uid, default_factory):
        factory = self._get_sku_config(self.trend_model_callable, uid)
        if factory is not None and not callable(factory):
            raise TypeError("trend_model_callable must be callable.")
        return (factory or default_factory)()

    def _get_residual_lags(self, uid, residual_part: np.ndarray) -> List[int]:
        config = self._get_sku_config(self.nlags, uid)
        if config == "auto":
            selected, _, _ = select_pami_lag(residual_part, **self.pami_params)
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

    def _fit_statsforecast(self, model, values, dates, uid):
        from statsforecast import StatsForecast

        frame = pd.DataFrame(
            {self.id_col_: uid, self.time_col_: dates, self.target_col_: values}
        )
        models = copy.deepcopy(model if isinstance(model, list) else [model])
        return StatsForecast(models=models, freq=self.freq_).fit(frame)

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

        residuals = []
        for uid, group in df.groupby(id_col):
            group = group.sort_values(time_col).copy()
            trend_model = self._get_trend_config(uid, partial(AutoETS, model="ZZN"))
            component = decomposed.loc[group.index]
            trend = component["trend"].to_numpy()
            residual = component["detrended"].to_numpy()
            lags = self._get_residual_lags(uid, residual)
            self.skus_nlags_[uid] = lags
            self.fitted_models_[uid] = {
                "trend": self._fit_statsforecast(
                    trend_model, trend, group[time_col], uid
                )
            }
            frame = group[[id_col, time_col] + self.exog_cols_].copy()
            frame[target_col] = residual
            residuals.append((uid, frame, lags))

        self._fit_residuals(residuals, prediction_intervals, static_features)
        return self

    def _stabilize(self, values, method, weight):
        if method == "hpi":
            return hpi(values, w_s=weight)
        if method == "hfi":
            return hfi(values, w_s=weight)
        raise ValueError("stabilization_method must be 'hpi' or 'hfi'.")

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
        predictions = []
        for uid, models in self.fitted_models_.items():
            trend_frame = models["trend"].predict(h=h)
            trend = (
                trend_frame[self._get_model_cols(trend_frame)].sum(axis=1).to_numpy()
            )
            frame = residual_predictions[
                residual_predictions[self.id_col_] == uid
            ].copy()
            for column in self._get_model_cols(frame):
                values = frame[column].to_numpy() + trend
                if self.log_transform:
                    values = np.expm1(values)
                if stabilization_method is not None and w_s > 0:
                    values = self._stabilize(values, stabilization_method, w_s)
                frame[column] = values
            predictions.append(frame)
        return pd.concat(predictions, ignore_index=True)
