# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import copy
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from tinyshift.forecasting.stabilization import hfi, hpi
from tinyshift.series.decomposition import extract_mstl_components
from tinyshift.series.forecastability import select_pami_lag
from tinyshift.series.seasonality import SeasonalPeriodDetector


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
        self.seasonal_detection_params = seasonal_detection_params
        self.trend_model_callable = trend_model_callable
        self.seasonal_model_callable = seasonal_model_callable
        self.nlags = nlags
        self.pami_params = pami_params
        self.log_transform = log_transform

    def _get_sku_config(self, config: Any, uid: Union[str, int]) -> Any:
        return config.get(uid) if isinstance(config, dict) else config

    def _get_model_cols(self, frame: pd.DataFrame) -> List[str]:
        return [
            column for column in frame if column not in (self.id_col_, self.time_col_)
        ]

    def _detect_panel_seasonal_periods(self, df: pd.DataFrame) -> None:
        """Detect periods once for every series configured as automatic."""
        auto_ids = [
            uid
            for uid in df[self.id_col_].unique()
            if self._get_sku_config(self.season_length, uid) == "auto"
        ]
        self.seasonal_detector_ = None
        if not auto_ids:
            return

        detection_df = df[df[self.id_col_].isin(auto_ids)]
        if self.log_transform:
            detection_df = detection_df.copy()
            detection_df[self.target_col_] = np.log1p(detection_df[self.target_col_])

        self.seasonal_detector_ = SeasonalPeriodDetector(
            **(self.seasonal_detection_params or {})
        ).fit(
            detection_df,
            id_col=self.id_col_,
            time_col=self.time_col_,
            target_col=self.target_col_,
        )

    def _get_raw_seasonal_periods(self, uid: Union[str, int]) -> List[int]:
        """Retrieve configured or previously detected periods for one SKU."""
        configured_periods = self._get_sku_config(self.season_length, uid)
        if configured_periods is None:
            raise ValueError(f"No season_length configured for unique_id {uid!r}.")

        if configured_periods == "auto":
            periods = (
                self.seasonal_detector_.results_.get(uid, {}).get(
                    "candidate_periods", []
                )
                if self.seasonal_detector_ is not None
                else []
            )
            if not periods:
                raise ValueError(
                    f"Could not automatically detect seasonal periods for unique_id {uid!r}. "
                    f"Consider passing an explicit 'fallback' in 'seasonal_detection_params' "
                    f"or a dictionary mapping in 'season_length' "
                    f"(e.g., season_length={{{uid!r}: 7}} or season_length={{{uid!r}: [7, 30]}})."
                )
            return periods

        if isinstance(configured_periods, int) and not isinstance(
            configured_periods, bool
        ):
            return [configured_periods]

        return list(configured_periods)

    def _validate_seasonal_periods_format(
        self, uid: Union[str, int], periods: List[int]
    ) -> List[int]:
        """Validate that seasonal periods are integers strictly greater than one."""
        if not periods or any(
            not isinstance(period, int) or isinstance(period, bool) or period <= 1
            for period in periods
        ):
            raise ValueError(
                f"season_length for unique_id {uid!r} must contain integer periods greater than one."
            )
        return sorted(set(periods))

    def _validate_series_length_for_mstl(
        self, uid: Union[str, int], series: np.ndarray, periods: List[int]
    ) -> None:
        """Validate that the series length satisfies MSTL minimum requirement."""
        max_p = max(periods)

        # For multiple seasonalities, MSTL requires a larger sample size to separate overlapping components
        min_required = 2 * sum(periods) if len(periods) > 1 else 2 * max_p

        if len(series) < min_required:
            raise ValueError(
                f"Series for unique_id {uid!r} has length {len(series)}, which is too short "
                f"for seasonal period {max_p} (MSTL requires at least {2 * max_p} observations). "
                f"Adjust your train window / step_size or set a smaller period for this SKU."
            )

    def _resolve_seasonal_periods(
        self, uid: Union[str, int], series: np.ndarray
    ) -> List[int]:
        """Resolve and validate the seasonal periods configured for one SKU."""
        raw_periods = self._get_raw_seasonal_periods(uid)
        periods = self._validate_seasonal_periods_format(uid, raw_periods)
        self._validate_series_length_for_mstl(uid, series, periods)
        return periods

    def _resolve_seasonal_factory(
        self, uid: Union[str, int], default_factory: Callable[[int], Any]
    ) -> Callable[[int], Any]:
        """Resolve the seasonal model factory configured for one SKU.

        The same ``default_factory`` reference is reused across every SKU that
        has no per-``unique_id`` override, which lets :meth:`fit` batch those
        series into a single panel-wide StatsForecast call.
        """
        factory = self._get_sku_config(self.seasonal_model_callable, uid)
        if factory is None:
            return default_factory
        if not callable(factory):
            raise TypeError(
                f"seasonal_model_callable for unique_id {uid!r} must be callable."
            )
        return factory

    def _get_residual_lags(
        self, uid: Union[str, int], residual_part: np.ndarray
    ) -> List[int]:
        """Calculate PAMI lags for one SKU, regardless of residual strategy."""
        configured_lags = self._get_sku_config(self.nlags, uid)
        if configured_lags == "auto":
            selected_lag, _, _ = select_pami_lag(
                residual_part, **(self.pami_params or {})
            )

            if selected_lag is None or (
                isinstance(selected_lag, (list, tuple, np.ndarray))
                and len(selected_lag) == 0
            ):
                raise ValueError(
                    f"Could not automatically select residual lag via PAMI for unique_id {uid!r}. "
                    f"Consider specifying explicit lags or a dictionary mapping in 'nlags' "
                    f"(e.g., nlags={{{uid!r}: 1}} or nlags={{{uid!r}: [1, 2, 3]}}), "
                    f"or adjusting 'pami_params'."
                )

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

    def _fit_panel(
        self,
        models: List[Any],
        rows: List[tuple],
    ) -> Any:
        """Fit one StatsForecast instance on a panel built from several series.

        Each row is ``(uid, values, dates)``. Every SKU still contributes its
        own values, but SKUs sharing the same resolved model factory are
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

    def _fit_statsforecast(
        self, models: Any, values: np.ndarray, dates: pd.Series, uid: Union[str, int]
    ) -> Any:
        """Fit one StatsForecast instance for a single SKU.

        Kept for backward compatibility; :meth:`fit` uses :meth:`_fit_panel`
        directly so that SKUs sharing a model factory can be batched together.
        """
        model_list = copy.deepcopy(models if isinstance(models, list) else [models])
        return self._fit_panel(model_list, [(uid, values, dates)])

    def _make_residual_frame(
        self, group: pd.DataFrame, residual_part: np.ndarray
    ) -> pd.DataFrame:
        frame = group[[self.id_col_, self.time_col_] + self.exog_cols_].copy()
        frame[self.target_col_] = residual_part
        return frame

    def _resolve_uid_config(
        self,
        uid: Union[str, int],
        values: np.ndarray,
        default_trend_factory: Callable[[], Any],
        default_seasonal_factory: Callable[[int], Any],
    ) -> tuple:
        """Resolve the trend factory, seasonal periods, and seasonal factory for one SKU."""
        trend_factory = (
            self._get_sku_config(self.trend_model_callable, uid)
            or default_trend_factory
        )
        if not callable(trend_factory):
            raise TypeError(
                f"trend_model_callable for unique_id {uid!r} must be callable."
            )
        periods = self._resolve_seasonal_periods(uid, values)
        seasonal_factory = self._resolve_seasonal_factory(uid, default_seasonal_factory)
        return trend_factory, periods, seasonal_factory

    def _decompose_uid(self, values: np.ndarray, periods: List[int]) -> tuple:
        """Run MSTL for one SKU and split the result into trend/seasonal/residual."""
        from statsmodels.tsa.seasonal import MSTL

        components = extract_mstl_components(
            MSTL(values, periods=periods).fit(), periods
        )
        trend = components["trend"].bfill().ffill().to_numpy()
        seasonal_cols = [
            column for column in components if column.startswith("seasonal")
        ]
        residual = components["resid"].fillna(0.0).to_numpy()
        return trend, seasonal_cols, components, residual

    def _register_trend_row(
        self,
        trend_groups: Dict[int, Dict[str, Any]],
        uid_trend_key: Dict[Union[str, int], int],
        trend_factory: Callable[[], Any],
        uid: Union[str, int],
        trend: np.ndarray,
        dates: pd.Series,
    ) -> None:
        """Assign one SKU's trend row to its shared-factory panel bucket."""
        trend_key = id(trend_factory)
        bucket = trend_groups.setdefault(
            trend_key, {"factory": trend_factory, "rows": []}
        )
        bucket["rows"].append((uid, trend, dates))
        uid_trend_key[uid] = trend_key

    def _register_seasonal_rows(
        self,
        seasonal_groups: Dict[tuple, Dict[str, Any]],
        uid_seasonal_keys: Dict[Union[str, int], List[tuple]],
        seasonal_factory: Callable[[int], Any],
        periods: List[int],
        seasonal_cols: List[str],
        components: pd.DataFrame,
        uid: Union[str, int],
        dates: pd.Series,
    ) -> None:
        """Assign one SKU's seasonal rows to their (period, factory) panel buckets."""
        seasonal_keys = []
        for period, column in zip(periods, seasonal_cols):
            key = (period, id(seasonal_factory))
            bucket = seasonal_groups.setdefault(
                key, {"factory": seasonal_factory, "period": period, "rows": []}
            )
            bucket["rows"].append(
                (uid, components[column].fillna(0.0).to_numpy(), dates)
            )
            seasonal_keys.append(key)
        uid_seasonal_keys[uid] = seasonal_keys

    def _fit_grouped_panels(
        self,
        trend_groups: Dict[int, Dict[str, Any]],
        seasonal_groups: Dict[tuple, Dict[str, Any]],
    ) -> tuple:
        """Fit one StatsForecast panel per trend/seasonal bucket."""
        trend_fitted = {
            key: self._fit_panel([bucket["factory"]()], bucket["rows"])
            for key, bucket in trend_groups.items()
        }
        seasonal_fitted = {
            key: self._fit_panel([bucket["factory"](bucket["period"])], bucket["rows"])
            for key, bucket in seasonal_groups.items()
        }
        return trend_fitted, seasonal_fitted

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

        if self.freq is None:
            raise ValueError(
                "Parameter 'freq' must be explicitly declared when initializing DMSTLWrapper."
            )
        self.freq_ = self.freq
        self.id_col_, self.time_col_, self.target_col_ = id_col, time_col, target_col
        self.exog_cols_ = [
            column for column in df if column not in (id_col, time_col, target_col)
        ]
        self._detect_panel_seasonal_periods(df)
        self.fitted_models_ = {}
        self.skus_nlags_ = {}
        residuals = []

        # Created once per fit() call so every SKU without a per-unique_id
        # override resolves to the SAME factory reference below, which lets
        # them be batched into one panel-wide StatsForecast call.
        default_trend_factory = partial(AutoETS, model="ZZN")

        def default_seasonal_factory(period: int) -> Any:
            try:
                return SeasonalNaive(
                    season_length=period, alias=f"SeasonalNaive-{period}"
                )
            except TypeError:
                return SeasonalNaive(season_length=period)

        trend_groups: Dict[int, Dict[str, Any]] = {}
        seasonal_groups: Dict[tuple, Dict[str, Any]] = {}
        uid_trend_key: Dict[Union[str, int], int] = {}
        uid_seasonal_keys: Dict[Union[str, int], List[tuple]] = {}

        for uid, group in df.groupby(id_col):
            group = group.sort_values(time_col).copy()
            values = group[target_col].to_numpy()

            if self.log_transform:
                values = np.log1p(values)

            trend_factory, periods, seasonal_factory = self._resolve_uid_config(
                uid, values, default_trend_factory, default_seasonal_factory
            )
            trend, seasonal_cols, components, residual = self._decompose_uid(
                values, periods
            )
            lags = self._get_residual_lags(uid, residual)
            self.skus_nlags_[uid] = lags

            dates = group[time_col]
            self._register_trend_row(
                trend_groups, uid_trend_key, trend_factory, uid, trend, dates
            )
            self._register_seasonal_rows(
                seasonal_groups,
                uid_seasonal_keys,
                seasonal_factory,
                periods,
                seasonal_cols,
                components,
                uid,
                dates,
            )

            residuals.append((uid, self._make_residual_frame(group, residual), lags))

        trend_fitted, seasonal_fitted = self._fit_grouped_panels(
            trend_groups, seasonal_groups
        )

        for uid in uid_trend_key:
            self.fitted_models_[uid] = {
                "trend": trend_fitted[uid_trend_key[uid]],
                "seasonal": [seasonal_fitted[key] for key in uid_seasonal_keys[uid]],
            }

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

    def _predict_component_for_uid(
        self,
        model: Any,
        cache: Dict[int, pd.DataFrame],
        uid: Union[str, int],
        h: int,
    ) -> np.ndarray:
        """Predict one shared trend/seasonal model once and return one SKU's values."""
        if id(model) not in cache:
            cache[id(model)] = model.predict(h=h)
        frame = cache[id(model)]
        frame = frame[frame[self.id_col_] == uid]
        return frame[self._get_model_cols(frame)].sum(axis=1).to_numpy()

    def _recombine_uid_forecast(
        self,
        frame: pd.DataFrame,
        trend: np.ndarray,
        seasonal: np.ndarray,
        stabilization_method: Optional[Literal["hpi", "hfi"]],
        w_s: float,
    ) -> pd.DataFrame:
        """Add trend/seasonal to each residual model column and post-process it."""
        for column in self._get_model_cols(frame):
            values = frame[column].to_numpy() + trend + seasonal
            if self.log_transform:
                values = np.expm1(values)
            if stabilization_method is not None and w_s > 0:
                values = self._stabilize(values, stabilization_method, w_s)
            frame[column] = values
        return frame

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
        # A trend/seasonal model may be shared by several SKUs (panel fit), so
        # each distinct fitted object is predicted only once here.
        trend_cache: Dict[int, pd.DataFrame] = {}
        seasonal_cache: Dict[int, pd.DataFrame] = {}
        predictions = []
        for uid, models in self.fitted_models_.items():
            trend = self._predict_component_for_uid(
                models["trend"], trend_cache, uid, h
            )
            seasonal = np.zeros(h)
            for model in models["seasonal"]:
                seasonal += self._predict_component_for_uid(
                    model, seasonal_cache, uid, h
                )
            frame = residual_predictions[
                residual_predictions[self.id_col_] == uid
            ].copy()
            frame = self._recombine_uid_forecast(
                frame, trend, seasonal, stabilization_method, w_s
            )
            predictions.append(frame)
        return pd.concat(predictions, ignore_index=True)
