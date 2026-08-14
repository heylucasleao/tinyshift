# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

import copy
from typing import Callable, Literal, Union, List, Tuple, Optional, Any, Dict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from tinyshift.utils.imports import requires_extra
from functools import partial


class DMSTLWrapper(BaseEstimator, RegressorMixin):
    """
    Decomposed Multiple Seasonal-Trend (DMSTL) local wrapper per unique_id.

    Decomposes multi-seasonal time series into trend, seasonal, and residual
    components using MSTL. Fits statistical base models on trend and seasonal
    components via StatsForecast, while modelling complex non-linear residual
    dynamics using MLForecast (which can hold single or multiple estimators).
    Optionally applies log-additive transformations (Box-Cox/Log-1p) and
    horizontal stabilization (HPI/HFI).

    Parameters
    ----------
    mf_resid : MLForecast
        Base MLForecast pipeline configured with one or multiple machine
        learning estimators (e.g., `models=[LGBMRegressor(), XGBRegressor()]`)
        to fit the residual component. Must have `freq` defined.
    season_length : int, list of int, or dict
        Seasonal period(s) passed directly to MSTL decomposition. A dictionary
        maps each unique identifier to an integer or list of integers.
    trend_model_callable : callable or dict of callable, optional
        Function without arguments that returns a configured StatsForecast
        trend model. A dictionary may map each unique identifier to its own
        callable. If None, an internal AutoETS(model="ZZN") model is used.
    seasonal_model_callable : callable or dict of callable, optional
        Function that receives a seasonal period and returns one configured
        StatsForecast model for that period. A dictionary may map each
        unique identifier to its own factory. If None, an internal factory
        creates one SeasonalNaive model for each period.
    log_transform : bool, default=False
        If True, applies np.log1p before decomposition and np.expm1 during
        prediction. Use this when the underlying series exhibits multiplicative
        dynamics (i.e., seasonal amplitudes or noise grow proportionally with
        the trend level). Log-transforming converts the multiplicative model
        (Y = T * S * R) into an additive space (log(Y) = log(T) + log(S) + log(R)),
        allowing standard MSTL to perform mathematically rigorous multiplicative
        decompositions while preventing negative domain errors.

    Attributes
    ----------
    season_length_ : list of int
        Formatted list of seasonal lengths.
    freq_ : str or int
        Effective time series frequency retrieved from `mf_resid.freq`.
    seasonal_models_ : dict
        Configured seasonal models by unique identifier and period.
    trend_models_ : dict
        Configured trend models by unique identifier.
    id_col_ : str
        Name of the unique identifier column set during fit.
    time_col_ : str
        Name of the timestamp/date column set during fit.
    target_col_ : str
        Name of the target variable column set during fit.
    exog_cols_ : list of str
        Names of exogenous feature columns identified during fit.
    fitted_models_ : dict
        Dictionary holding fitted StatsForecast (trend, seasonal) and
        MLForecast (residual) models mapped by unique_id.

    Notes
    -----
    **Horizontal Stabilization (HPI / HFI):**
    Post-processing horizontal stabilization routines can be enabled during `predict()`
    via `stabilization_method` to mitigate variance explosion across multi-step horizons:

    - **HPI (Horizontal Penalization Invariant):** Applies a penalization penalty to
      abrupt step-to-step variations in the multi-step horizon. Use when forecasts
      suffer from high variance or non-physical oscillations across adjacent time steps.
    - **HFI (Horizontal Filtering Invariant):** Filters high-frequency noise from
      the forecasted path while preserving underlying momentum. Use when multi-step
      predictions are overly noisy or sensitive to residual ML fluctuations.
    """

    @requires_extra("series")
    def __init__(
        self,
        mf_resid: Any,
        season_length: Union[
            int, List[int], Dict[Union[str, int], Union[int, List[int]]]
        ],
        trend_model_callable: Optional[
            Union[
                Callable[[], Any],
                Dict[Union[str, int], Callable[[], Any]],
            ]
        ] = None,
        seasonal_model_callable: Optional[
            Union[
                Callable[[int], Any],
                Dict[Union[str, int], Callable[[int], Any]],
            ]
        ] = None,
        log_transform: bool = False,
    ) -> None:
        self.mf_resid = mf_resid
        self.season_length = season_length
        self.trend_model_callable = trend_model_callable
        self.seasonal_model_callable = seasonal_model_callable
        self.log_transform = log_transform

    def _get_sku_config(self, config, uid: Union[str, int]):
        """Resolve a global or per-series configuration value."""
        if isinstance(config, dict):
            return config.get(uid)
        return config

    def _extract_freq(self) -> Union[str, int]:
        """
        Extract and validate the frequency from the MLForecast instance.

        Returns
        -------
        freq : str or int
            Frequency set in the MLForecast instance.

        Raises
        ------
        ValueError
            If `mf_resid.freq` is missing or None.
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

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing time series predictions.

        Returns
        -------
        cols : list of str
            Columns representing model outputs (excluding id and time columns).
        """
        return [c for c in df.columns if c not in [self.id_col_, self.time_col_]]

    def _process_components(
        self, components_df: pd.DataFrame, split_seasonal: bool = False
    ) -> Tuple[np.ndarray, Union[np.ndarray, List[np.ndarray]], np.ndarray]:
        """
        Impute missing values and aggregate trend, seasonal, and residual signals.

        Parameters
        ----------
        components_df : pd.DataFrame
            DataFrame containing raw MSTL components.

        Returns
        -------
        trend_part : ndarray of shape (n_samples,)
            Backfilled and forward-filled trend values.
        seasonal_part : ndarray or list of ndarray
            Summed seasonal component across all seasonal channels, or each
            seasonal channel separately when ``split_seasonal=True``.
        residual_part : ndarray of shape (n_samples,)
            Zero-filled residual sequence.
        """
        trend_part = components_df["trend"].bfill().ffill().values
        seasonal_cols = [c for c in components_df.columns if c.startswith("seasonal")]
        if split_seasonal:
            seasonal_part = [
                components_df[column].fillna(0.0).values for column in seasonal_cols
            ]
        else:
            seasonal_part = components_df[seasonal_cols].sum(axis=1).values
        residual_part = components_df["resid"].fillna(0.0).values
        return trend_part, seasonal_part, residual_part

    def _get_trend_config(
        self,
        uid: Union[str, int],
        default_trend_callable: Callable[[], Any],
    ) -> Any:
        """
        Resolve the trend model configured for a SKU.

        Parameters
        ----------
        uid : str or int
            Unique identifier of the series.
        default_trend_callable : callable
            Factory without arguments that creates the fallback trend model.

        Returns
        -------
        trend_model : StatsForecast model
            A new model instance created by the configured or fallback factory.

        Raises
        ------
        TypeError
            If `trend_model_callable` resolves to a non-callable value.
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

    def _get_seasonal_config(
        self, uid: Union[str, int], seasonal_naive_model
    ) -> Tuple[List[int], List[Any]]:
        """
        Resolve the seasonal periods and models configured for a SKU.

        Parameters
        ----------
        uid : str or int
            Unique identifier of the series.
        seasonal_naive_model : callable
            SeasonalNaive constructor used by the default factory.

        Returns
        -------
        season_lengths : list of int
            Seasonal periods normalized as a list.
        seasonal_models : list
            Seasonal models normalized as a list, with one model per period.

        Raises
        ------
        ValueError
            If the SKU has no configured periods or if a period is not a
            positive integer greater than one.
        """
        season_length = self._get_sku_config(self.season_length, uid)
        if season_length is None:
            raise ValueError(
                f"No season_length configured for unique_id {uid!r}. "
                "Provide a period for every series."
            )

        season_lengths = (
            [season_length] if isinstance(season_length, int) else season_length
        )
        if (
            not isinstance(season_lengths, list)
            or not season_lengths
            or len(set(season_lengths)) != len(season_lengths)
            or any(
                not isinstance(period, int) or isinstance(period, bool) or period <= 1
                for period in season_lengths
            )
        ):
            raise ValueError(
                f"season_length for unique_id {uid!r} must contain positive "
                "integer periods greater than one without duplicates."
            )

        seasonal_model_callable = self._get_sku_config(
            self.seasonal_model_callable, uid
        )
        if seasonal_model_callable is None:

            def seasonal_model_callable(period):
                try:
                    return seasonal_naive_model(
                        season_length=period,
                        alias=f"SeasonalNaive-{period}",
                    )
                except TypeError as error:
                    if "alias" not in str(error):
                        raise
                    return seasonal_naive_model(season_length=period)

            seasonal_models = [
                seasonal_model_callable(period) for period in season_lengths
            ]
        else:
            if not callable(seasonal_model_callable):
                raise TypeError(
                    f"seasonal_model_callable for unique_id {uid!r} must be "
                    "a callable that accepts one seasonal period."
                )
            seasonal_models = [
                seasonal_model_callable(period) for period in season_lengths
            ]

        return season_lengths, seasonal_models

    def _fit_statsforecast(
        self,
        models,
        values: np.ndarray,
        dates: pd.Series,
        uid: Union[str, int],
        freq: Union[str, int],
    ) -> Any:
        """
        Fit a StatsForecast pipeline on a single component series.

        Parameters
        ----------
        models : StatsForecast model or list of models
            Statistical forecasting model(s) to fit.
        values : ndarray of shape (n_samples,)
            Target component values (trend or seasonal).
        dates : pd.Series
            Datetime or step index corresponding to the target values.
        uid : str or int
            Unique identifier for the group.
        freq : str or int
            Series frequency.

        Returns
        -------
        sf : StatsForecast
            Fitted StatsForecast instance.
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
        Fit a isolated copy of the base MLForecast pipeline on the extracted residual component.

        Parameters
        ----------
        group : pd.DataFrame
            Group-level source DataFrame containing time and exogenous features.
        residual_part : ndarray of shape (n_samples,)
            Extracted residual values to be modeled.
        prediction_intervals : PredictionIntervals, optional
            Configuration for conformal prediction intervals.

        Returns
        -------
        fitted_mf : MLForecast
            Fitted deep copy of the base MLForecast instance.
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
    ) -> "DMSTLWrapper":
        """
        Fit MSTL decomposition and sub-models for each unique group in the data.

        Parameters
        ----------
        df : pd.DataFrame
            Long-format panel dataset containing time series and exogenous features.
        id_col : str, default="unique_id"
            Column identifying individual time series.
        time_col : str, default="ds"
            Column identifying timestamps or integer time steps.
        target_col : str, default="y"
            Target variable column name.
        prediction_intervals : PredictionIntervals, optional
            MLForecast conformal interval configuration for residual uncertainty estimation.

        Returns
        -------
        self : DMSTLWrapper
            Fitted estimator.

        Raises
        ------
        ValueError
            If `mf_resid.freq` is missing or invalid.
        """
        from statsmodels.tsa.seasonal import MSTL
        from statsforecast.models import AutoETS, SeasonalNaive
        from tinyshift.series import extract_mstl_components

        self.season_length_ = (
            self.season_length
            if isinstance(self.season_length, dict)
            else (
                [self.season_length]
                if isinstance(self.season_length, int)
                else self.season_length
            )
        )
        self.freq_ = self._extract_freq()

        self.id_col_ = id_col
        self.time_col_ = time_col
        self.target_col_ = target_col
        self.exog_cols_ = [
            c for c in df.columns if c not in [id_col, time_col, target_col]
        ]
        self.fitted_models_ = {}
        self.seasonal_models_ = {}
        self.trend_models_ = {}

        for uid, group in df.groupby(id_col):
            trend_model = self._get_trend_config(uid, partial(AutoETS, model="ZZN"))

            season_lengths, seasonal_models = self._get_seasonal_config(
                uid, SeasonalNaive
            )
            group_sorted = group.sort_values(time_col).copy()
            y_series = group_sorted[target_col].values

            if self.log_transform:
                y_series = np.log1p(y_series)

            dates = group_sorted[time_col]

            mstl = MSTL(y_series, periods=season_lengths)
            res = mstl.fit()
            components_df = extract_mstl_components(res, season_lengths)
            trend_part, seasonal_parts, residual_part = self._process_components(
                components_df, split_seasonal=True
            )

            sf_trend = self._fit_statsforecast(
                trend_model, trend_part, dates, uid, self.freq_
            )
            sf_seasonal = [
                self._fit_statsforecast(
                    seasonal_model,
                    seasonal_part,
                    dates,
                    uid,
                    self.freq_,
                )
                for seasonal_model, seasonal_part in zip(
                    seasonal_models, seasonal_parts
                )
            ]
            fitted_mf = self._fit_mlforecast(
                group_sorted, residual_part, prediction_intervals
            )

            self.fitted_models_[uid] = {
                "trend": sf_trend,
                "seasonal": sf_seasonal,
                "residual": fitted_mf,
            }
            self.seasonal_models_[uid] = seasonal_models
            self.trend_models_[uid] = trend_model

        return self

    def _apply_horizontal_stabilization(
        self, y_hat: np.ndarray, method: Literal["hpi", "hfi"], w_s: float
    ) -> np.ndarray:
        """
        Apply horizontal penalization/filtering stabilization to raw predictions.

        Parameters
        ----------
        y_hat : ndarray of shape (h,)
            Raw recombined forecasts.
        method : {"hpi", "hfi"}
            Stabilization algorithm: 'hpi' (Horizontal Penalization Invariant) or
            'hfi' (Horizontal Filtering Invariant).
        w_s : float
            Stabilization smoothing intensity weight.

        Returns
        -------
        y_hat_stable : ndarray of shape (h,)
            Stabilized prediction sequence.

        Raises
        ------
        ValueError
            If method is not 'hpi' or 'hfi'.
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
        Generate future forecasts by recombining trend, seasonal, and residual predictions.

        Parameters
        ----------
        h : int
            Forecast horizon (number of steps ahead to predict).
        X_df : pd.DataFrame, optional
            DataFrame containing exogenous features for the forecast horizon.
        level : list of int or float, optional
            Confidence levels (0-100) for prediction intervals (e.g., [80, 95]).
        stabilization_method : {"hpi", "hfi"}, optional
            Horizontal stabilization technique to post-process point and interval forecasts.
        w_s : float, default=0.0
            Stabilization weight parameter. Active only if `w_s > 0.0` and a method is selected.

        Returns
        -------
        preds_df : pd.DataFrame
            DataFrame with predictions from all MLForecast estimators, unique IDs, timestamps,
            and optional prediction interval columns.

        Raises
        ------
        RuntimeError
            If model is not fitted prior to calling predict.

        Notes
        -----
        - Recombination occurs in log space if `log_transform=True`, followed by scale inversion (`expm1`).
        - Trend and seasonal projections are added across all model and interval columns generated by MLForecast.
        """
        if not hasattr(self, "fitted_models_") or not self.fitted_models_:
            raise RuntimeError(
                "The model must be fitted with .fit() before making predictions."
            )

        preds_list = []

        for uid, models in self.fitted_models_.items():
            sf_trend = models["trend"]
            sf_seasonal = models["seasonal"]
            mf_resid = models["residual"]

            df_trend = sf_trend.predict(h=h)
            trend_cols = self._get_model_cols(df_trend)
            trend_preds = df_trend[trend_cols].sum(axis=1).values
            seasonal_preds = np.zeros(h)
            for seasonal_model in sf_seasonal:
                df_seasonal = seasonal_model.predict(h=h)
                seasonal_cols = self._get_model_cols(df_seasonal)
                seasonal_preds += df_seasonal[seasonal_cols].sum(axis=1).values

            X_uid = X_df[X_df[self.id_col_] == uid].copy() if X_df is not None else None
            df_resid = mf_resid.predict(h=h, X_df=X_uid, level=level).copy()

            model_cols = self._get_model_cols(df_resid)

            for col in model_cols:
                res = df_resid[col].values + trend_preds + seasonal_preds

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
