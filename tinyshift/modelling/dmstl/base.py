# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import copy
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from tinyshift.series import detect_seasonal_periods, extract_mstl_components, hfi, hpi
from tinyshift.series import select_pami_lag


class BaseDMSTL(BaseEstimator, RegressorMixin):
    """Shared decomposition and recombination workflow for DMSTL strategies.

    This class owns the common panel workflow: it resolves seasonal periods,
    decomposes each series with MSTL, fits trend and seasonal components with
    StatsForecast, selects residual lags with PAMI, and recombines forecasts.
    Concrete strategies implement only how residual frames are fitted and
    predicted through :meth:`_fit_residuals` and :meth:`_predict_residuals`.

    The class is intended to be used through :class:`DMSTLLocalWrapper`,
    :class:`DMSTLGlobalWrapper`, or the public :class:`DMSTLWrapper` facade.
    """

    def __init__(
        self,
        residual_model_callable: Optional[
            Union[Callable[..., Any], Dict[Union[str, int], Callable[..., Any]]]
        ] = None,
        freq: Optional[Union[str, int]] = None,
        season_length: Any = "auto",
        seasonal_detection_params: Optional[Dict[str, Any]] = None,
        trend_model_callable: Optional[
            Union[Callable[[], Any], Dict[Union[str, int], Callable[[], Any]]]
        ] = None,
        seasonal_model_callable: Optional[
            Union[Callable[[int], Any], Dict[Union[str, int], Callable[[int], Any]]]
        ] = None,
        nlags: Any = "auto",
        pami_params: Optional[Dict[str, Any]] = None,
        log_transform: bool = False,
    ) -> None:
        self.residual_model_callable = residual_model_callable
        self.freq = freq
        self.season_length = season_length
        self.seasonal_detection_params = seasonal_detection_params or {}
        self.trend_model_callable = trend_model_callable
        self.seasonal_model_callable = seasonal_model_callable
        self.nlags = nlags
        self.pami_params = pami_params or {}
        self.log_transform = log_transform

    def _get_sku_config(self, config: Any, uid: Union[str, int]) -> Any:
        return config.get(uid) if isinstance(config, dict) else config

    def _get_model_cols(self, frame: pd.DataFrame) -> List[str]:
        return [
            column for column in frame if column not in (self.id_col_, self.time_col_)
        ]

    def _get_seasonal_config(
        self, uid: Union[str, int], series: np.ndarray, seasonal_naive_model: Any
    ) -> tuple[List[int], List[Any]]:
        configured_periods = self._get_sku_config(self.season_length, uid)
        if configured_periods is None:
            raise ValueError(f"No season_length configured for unique_id {uid!r}.")
        if configured_periods == "auto":
            periods = detect_seasonal_periods(series, **self.seasonal_detection_params)
        elif isinstance(configured_periods, int) and not isinstance(
            configured_periods, bool
        ):
            periods = [configured_periods]
        else:
            periods = list(configured_periods)
        if not periods or any(
            not isinstance(period, int) or isinstance(period, bool) or period <= 1
            for period in periods
        ):
            raise ValueError(
                f"season_length for unique_id {uid!r} must contain integer periods greater than one."
            )
        periods = sorted(set(periods))

        factory = self._get_sku_config(self.seasonal_model_callable, uid)
        if factory is None:

            def factory(period: int) -> Any:
                try:
                    return seasonal_naive_model(
                        season_length=period, alias=f"SeasonalNaive-{period}"
                    )
                except TypeError:
                    return seasonal_naive_model(season_length=period)

        if not callable(factory):
            raise TypeError(
                f"seasonal_model_callable for unique_id {uid!r} must be callable."
            )
        return periods, [factory(period) for period in periods]

    def _get_residual_lags(
        self, uid: Union[str, int], residual_part: np.ndarray
    ) -> List[int]:
        """Calculate PAMI lags for one SKU, regardless of residual strategy."""
        configured_lags = self._get_sku_config(self.nlags, uid)
        if configured_lags == "auto":
            selected_lag, _, _ = select_pami_lag(residual_part, **self.pami_params)
            if isinstance(selected_lag, int):
                return [max(selected_lag, 1)]
            return list(selected_lag) or [1]
        if isinstance(configured_lags, int) and configured_lags > 0:
            return list(range(1, configured_lags + 1))
        if isinstance(configured_lags, list) and configured_lags:
            return configured_lags
        raise ValueError(
            f"Invalid lags configuration for unique_id {uid!r}: {configured_lags!r}"
        )

    def _fit_statsforecast(
        self, models: Any, values: np.ndarray, dates: pd.Series, uid: Union[str, int]
    ) -> Any:
        from statsforecast import StatsForecast

        frame = pd.DataFrame(
            {self.id_col_: uid, self.time_col_: dates, self.target_col_: values}
        )
        model_list = copy.deepcopy(models if isinstance(models, list) else [models])
        return StatsForecast(models=model_list, freq=self.freq_).fit(frame)

    def _make_residual_frame(
        self, group: pd.DataFrame, residual_part: np.ndarray
    ) -> pd.DataFrame:
        frame = group[[self.id_col_, self.time_col_] + self.exog_cols_].copy()
        frame[self.target_col_] = residual_part
        return frame

    def _fit_residuals(
        self,
        residuals: List[tuple[Union[str, int], pd.DataFrame, List[int]]],
        prediction_intervals: Optional[Any],
        static_features: Optional[List[str]],
    ) -> None:
        raise NotImplementedError

    def _predict_residuals(
        self,
        h: int,
        X_df: Optional[pd.DataFrame],
        level: Optional[List[Union[int, float]]],
    ) -> pd.DataFrame:
        raise NotImplementedError

    def fit(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        prediction_intervals: Optional[Any] = None,
        static_features: Optional[List[str]] = None,
    ) -> "BaseDMSTL":
        from statsforecast.models import AutoETS, SeasonalNaive
        from statsmodels.tsa.seasonal import MSTL

        if self.freq is None:
            raise ValueError(
                "Parameter 'freq' must be explicitly declared when initializing DMSTLWrapper."
            )
        self.freq_ = self.freq
        self.id_col_, self.time_col_, self.target_col_ = id_col, time_col, target_col
        self.exog_cols_ = [
            column for column in df if column not in (id_col, time_col, target_col)
        ]
        self.fitted_models_ = {}
        self.skus_nlags_ = {}
        residuals = []

        for uid, group in df.groupby(id_col):
            group = group.sort_values(time_col).copy()
            values = group[target_col].to_numpy()

            if self.log_transform:
                values = np.log1p(values)

            trend_factory = self._get_sku_config(
                self.trend_model_callable, uid
            ) or partial(AutoETS, model="ZZN")

            if not callable(trend_factory):
                raise TypeError(
                    f"trend_model_callable for unique_id {uid!r} must be callable."
                )

            periods, seasonal_models = self._get_seasonal_config(
                uid, values, SeasonalNaive
            )

            components = extract_mstl_components(
                MSTL(values, periods=periods).fit(), periods
            )
            trend = components["trend"].bfill().ffill().to_numpy()
            seasonal = [
                components[column].fillna(0.0).to_numpy()
                for column in components
                if column.startswith("seasonal")
            ]
            residual = components["resid"].fillna(0.0).to_numpy()
            lags = self._get_residual_lags(uid, residual)
            self.skus_nlags_[uid] = lags
            self.fitted_models_[uid] = {
                "trend": self._fit_statsforecast(
                    trend_factory(), trend, group[time_col], uid
                ),
                "seasonal": [
                    self._fit_statsforecast(model, component, group[time_col], uid)
                    for model, component in zip(seasonal_models, seasonal)
                ],
            }
            residuals.append((uid, self._make_residual_frame(group, residual), lags))

        self._fit_residuals(residuals, prediction_intervals, static_features)
        return self

    def _stabilize(
        self, values: np.ndarray, method: Literal["hpi", "hfi"], weight: float
    ) -> np.ndarray:
        if method == "hpi":
            return hpi(values, w_s=weight)
        if method == "hfi":
            return hfi(values, w_s=weight)
        raise ValueError("stabilization_method must be 'hpi' or 'hfi'.")

    def predict(
        self,
        h: int,
        X_df: Optional[pd.DataFrame] = None,
        level: Optional[List[Union[int, float]]] = None,
        stabilization_method: Optional[Literal["hpi", "hfi"]] = None,
        w_s: float = 0.0,
    ) -> pd.DataFrame:
        if not hasattr(self, "fitted_models_"):
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
            seasonal = np.zeros(h)
            for model in models["seasonal"]:
                seasonal_frame = model.predict(h=h)
                seasonal += (
                    seasonal_frame[self._get_model_cols(seasonal_frame)]
                    .sum(axis=1)
                    .to_numpy()
                )
            frame = residual_predictions[
                residual_predictions[self.id_col_] == uid
            ].copy()
            for column in self._get_model_cols(frame):
                values = frame[column].to_numpy() + trend + seasonal
                if self.log_transform:
                    values = np.expm1(values)
                if stabilization_method is not None and w_s > 0:
                    values = self._stabilize(values, stabilization_method, w_s)
                frame[column] = values
            predictions.append(frame)
        return pd.concat(predictions, ignore_index=True)
