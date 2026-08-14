# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

import copy
from typing import Callable, Literal, Union, List, Optional, Any, Dict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from tinyshift.utils.imports import requires_extra
from functools import partial


class DTLWrapper(BaseEstimator, RegressorMixin):
    """
    Decomposed Trend-Local (DTL) wrapper per unique_id for non-seasonal data.

    Decomposes non-seasonal time series into trend and residual components using
    robust LOWESS smoothing. Fits a statistical base model on the trend component
    via StatsForecast, while modelling complex non-linear residual dynamics using
    MLForecast (which can hold single or multiple estimators). Optionally applies
    log-additive transformations (Box-Cox/Log-1p) and horizontal stabilization (HPI/HFI).

    Parameters
    ----------
    mf_resid : MLForecast
        Base MLForecast pipeline configured with one or multiple machine
        learning estimators (e.g., `models=[LGBMRegressor(), XGBRegressor()]`)
        to fit the residual component. Must have `freq` defined.
    trend_model_callable : callable or dict of callable, optional
        Function without arguments that returns a configured StatsForecast
        trend model. A dictionary may map each unique identifier to its own
        callable. If None, an internal AutoETS(model="ZZN") model is used.
    trend_frac : float, default=0.2
        The fraction of the data used when estimating each y-value in the LOWESS
        trend smoothing. Higher values result in smoother trends.
    robust : bool, default=True
        If True, applies robustifying iterations to the LOWESS algorithm,
        down-weighting outliers during trend extraction.
    log_transform : bool, default=False
        If True, applies np.log1p before decomposition and np.expm1 during
        prediction. Use this when the underlying series exhibits multiplicative
        dynamics.

    Attributes
    ----------
    freq_ : str or int
        Effective time series frequency retrieved from `mf_resid.freq`.
    id_col_ : str
        Name of the unique identifier column set during fit.
    time_col_ : str
        Name of the timestamp/date column set during fit.
    target_col_ : str
        Name of the target variable column set during fit.
    exog_cols_ : list of str
        Names of exogenous feature columns identified during fit.
    fitted_models_ : dict
        Dictionary holding fitted StatsForecast (trend) and
        MLForecast (residual) models mapped by unique_id.
    """

    @requires_extra("series")
    def __init__(
        self,
        mf_resid: Any,
        trend_model_callable: Optional[
            Union[
                Callable[[], Any],
                Dict[Union[str, int], Callable[[], Any]],
            ]
        ] = None,
        trend_frac: float = 0.2,
        robust: bool = True,
        log_transform: bool = False,
    ) -> None:
        self.mf_resid = mf_resid
        self.trend_model_callable = trend_model_callable
        self.trend_frac = trend_frac
        self.robust = robust
        self.log_transform = log_transform

    def _get_sku_config(self, config, uid: Union[str, int]):
        """Resolve a global or per-series configuration value."""
        if isinstance(config, dict):
            return config.get(uid)
        return config

    def _extract_freq(self) -> Union[str, int]:
        """
        Extract and validate the frequency from the MLForecast instance.
        """
        freq = getattr(self.mf_resid, "freq", None)
        if freq is None:
            raise ValueError(
                "The provided MLForecast instance does not have 'freq' defined. "
                "Ensure 'freq' is set when instantiating MLForecast."
            )
        return freq

    def _get_model_cols(self, df: pd.DataFrame) -> List[str]:
        """
        Extract model prediction column names from a forecasted DataFrame.
        """
        return [c for c in df.columns if c not in [self.id_col_, self.time_col_]]

    def _get_trend_config(
        self,
        uid: Union[str, int],
        default_trend_callable: Callable[[], Any],
    ) -> Any:
        """
        Resolve the trend model configured for a SKU.
        """
        trend_model_callable = self._get_sku_config(self.trend_model_callable, uid)
        if trend_model_callable is not None:
            if not callable(trend_model_callable):
                raise TypeError(
                    f"trend_model_callable for unique_id {uid!r} must be "
                    "a callable that accepts no arguments."
                )
            return trend_model_callable()

        return default_trend_callable()

    def _fit_statsforecast(
        self,
        models,
        values: np.ndarray,
        dates: pd.Series,
        uid: Union[str, int],
        freq: Union[str, int],
    ) -> Any:
        """
        Fit a StatsForecast pipeline on the trend component series.
        """
        from statsforecast import StatsForecast

        sf_df = pd.DataFrame(
            {self.id_col_: uid, self.time_col_: dates, self.target_col_: values}
        )
        models_list = copy.deepcopy(models if isinstance(models, list) else [models])
        return StatsForecast(models=models_list, freq=freq).fit(sf_df)

    def _fit_mlforecast(
        self, group: pd.DataFrame, residual_part: np.ndarray, prediction_intervals=None
    ) -> Any:
        """
        Fit an isolated copy of the base MLForecast pipeline on the extracted residual component.
        """
        df_residual = group[[self.id_col_, self.time_col_] + self.exog_cols_].copy()
        df_residual[self.target_col_] = residual_part

        mf_resid = copy.deepcopy(self.mf_resid)
        mf_resid.fit(
            df_residual,
            id_col=self.id_col_,
            time_col=self.time_col_,
            target_col=self.target_col_,
            prediction_intervals=prediction_intervals,
        )
        return mf_resid

    @requires_extra("series")
    def fit(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        prediction_intervals=None,
    ) -> "DTLWrapper":
        """
        Fit Trend decomposition and sub-models for each unique group in the data.
        """
        from statsforecast.models import AutoETS
        from statsmodels.nonparametric.smoothers_lowess import lowess

        self.freq_ = self._extract_freq()

        self.id_col_ = id_col
        self.time_col_ = time_col
        self.target_col_ = target_col
        self.exog_cols_ = [
            c for c in df.columns if c not in [id_col, time_col, target_col]
        ]
        self.fitted_models_ = {}

        for uid, group in df.groupby(id_col):
            trend_model = self._get_trend_config(uid, partial(AutoETS, model="ZZN"))

            group_sorted = group.sort_values(time_col).copy()
            y_series = group_sorted[target_col].values

            if self.log_transform:
                y_series = np.log1p(y_series)

            dates = group_sorted[time_col]

            y_series_clean = (
                pd.Series(y_series)
                .interpolate(method="linear", limit_direction="both")
                .values
            )

            trend_part = lowess(
                y_series_clean,
                np.arange(len(y_series_clean)),
                frac=self.trend_frac,
                it=3 if self.robust else 0,
                return_sorted=False,
            )
            residual_part = y_series - trend_part

            sf_trend = self._fit_statsforecast(
                trend_model, trend_part, dates, uid, self.freq_
            )
            fitted_mf = self._fit_mlforecast(
                group_sorted, residual_part, prediction_intervals
            )

            self.fitted_models_[uid] = {
                "trend": sf_trend,
                "residual": fitted_mf,
            }

        return self

    def _apply_horizontal_stabilization(
        self, y_hat: np.ndarray, method: Literal["hpi", "hfi"], w_s: float
    ) -> np.ndarray:
        """
        Apply horizontal penalization/filtering stabilization to raw predictions.
        """
        from tinyshift.series import hpi, hfi

        if method == "hpi":
            return hpi(y_hat, w_s=w_s)
        elif method == "hfi":
            return hfi(y_hat, w_s=w_s)
        else:
            raise ValueError(
                f"Invalid method '{method}'. Choose either 'hpi' or 'hfi'."
            )

    @requires_extra("series")
    def predict(
        self,
        h: int,
        X_df: Optional[pd.DataFrame] = None,
        level: Optional[List[Union[int, float]]] = None,
        stabilization_method: Optional[Literal["hpi", "hfi"]] = None,
        w_s: float = 0.0,
    ) -> pd.DataFrame:
        """
        Generate future forecasts by recombining trend and residual predictions.
        """
        if not hasattr(self, "fitted_models_") or not self.fitted_models_:
            raise RuntimeError(
                "The model must be fitted with .fit() before making predictions."
            )

        preds_list = []

        for uid, models in self.fitted_models_.items():
            sf_trend = models["trend"]
            mf_resid = models["residual"]

            df_trend = sf_trend.predict(h=h)
            trend_cols = self._get_model_cols(df_trend)
            trend_preds = df_trend[trend_cols].sum(axis=1).values

            X_uid = X_df[X_df[self.id_col_] == uid].copy() if X_df is not None else None
            df_resid = mf_resid.predict(h=h, X_df=X_uid, level=level).copy()

            model_cols = self._get_model_cols(df_resid)

            for col in model_cols:
                res = df_resid[col].values + trend_preds

                if self.log_transform:
                    res = np.expm1(res)

                if stabilization_method is not None and w_s > 0.0:
                    res = self._apply_horizontal_stabilization(
                        y_hat=res, method=stabilization_method, w_s=w_s
                    )

                df_resid[col] = res

            preds_list.append(df_resid)

        preds_df = pd.concat(preds_list, ignore_index=True)
        return preds_df
