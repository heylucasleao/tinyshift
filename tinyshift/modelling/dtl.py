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
    residual_model_callable : callable or dict of callable, optional
        Factory that receives ``nlags`` and ``freq`` and returns the MLForecast
        model used for the residual component. A dictionary may map each
        unique identifier to its own factory. The factory may accept these
        arguments as keywords or as two positional arguments.
    freq : str or int
        Frequency passed to the residual model and StatsForecast models.
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
    nlags : int, list of int, dict, "auto", or None, default="auto"
        Residual lag configuration. An integer creates all lags from 1 through
        that value; a list is used directly; ``"auto"`` selects one lag with
        :func:`tinyshift.series.select_pami_lag`. A dictionary may map each
        unique identifier to one of these configurations.
    pami_params : dict, optional
        Keyword arguments forwarded to ``select_pami_lag`` when ``nlags`` is
        ``"auto"``.

    Notes
    -----
    When ``nlags="auto"``, the wrapper uses
    :func:`tinyshift.series.select_pami_lag` to select a residual lag from the
    first local minimum of permutation auto-mutual information (PAMI), falling
    back to the lowest evaluated value when no local minimum exists.

    Examples
    --------
    A residual model factory receives the selected lags and frequency::

        from mlforecast import MLForecast
        from sklearn.ensemble import RandomForestRegressor

        def residual_model(nlags, freq):
            return MLForecast(
                models=[RandomForestRegressor(n_estimators=100, random_state=0)],
                lags=nlags,
                freq=freq,
            )

        model = DTLWrapper(
            residual_model_callable=residual_model,
            freq="D",
            nlags="auto",
            pami_params={"max_tau": 48, "m": 3, "delay": 1},
        )

    Manual lags and per-series configuration are also supported::

        model = DTLWrapper(
            residual_model_callable=residual_model,
            freq="D",
            nlags={"series_a": [1, 2, 3], "series_b": "auto"},
            trend_frac=0.3,
            robust=True,
        )

    Attributes
    ----------
    freq_ : str or int
        Effective time series frequency configured through ``freq``.
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
        residual_model_callable: Optional[
            Union[
                Callable[[List[int], Union[str, int]], Any],
                Dict[Union[str, int], Callable[[List[int], Union[str, int]], Any]],
            ]
        ] = None,
        freq: Optional[Union[str, int]] = None,
        trend_model_callable: Optional[
            Union[
                Callable[[], Any],
                Dict[Union[str, int], Callable[[], Any]],
            ]
        ] = None,
        trend_frac: float = 0.2,
        robust: bool = True,
        log_transform: bool = False,
        nlags: Optional[
            Union[
                int,
                List[int],
                Dict[Union[str, int], Union[int, List[int], Literal["auto"]]],
                Literal["auto"],
            ]
        ] = "auto",
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

    def _get_sku_config(self, config, uid: Union[str, int]):
        """
        Resolve a global or per-series configuration value.

        Parameters
        ----------
        config : object or dict
            Global configuration value or mapping keyed by unique identifier.
        uid : str or int
            Unique identifier of the series.

        Returns
        -------
        object
            The per-series value when ``config`` is a dictionary, otherwise
            the global configuration value.
        """
        if isinstance(config, dict):
            return config.get(uid)
        return config

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
            Columns representing model outputs, excluding identifier and time
            columns.
        """
        return [c for c in df.columns if c not in [self.id_col_, self.time_col_]]

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

    def _get_residual_lags(
        self,
        uid: Union[str, int],
        residual_part: np.ndarray,
    ) -> List[int]:
        """Resolve manual or PAMI-selected residual lags for one series."""
        lags_config = self._get_sku_config(self.nlags, uid)

        if lags_config == "auto":
            from tinyshift.series import select_pami_lag

            selected_lag, _, _ = select_pami_lag(
                residual_part, **self.pami_params, return_mode="value_only"
            )
            return [selected_lag] if selected_lag > 0 else [1]

        if isinstance(lags_config, int) and not isinstance(lags_config, bool):
            if lags_config < 1:
                raise ValueError("nlags must be positive")
            return list(range(1, lags_config + 1))

        if isinstance(lags_config, list):
            return lags_config

        raise ValueError(
            f"Invalid lags configuration for unique_id {uid!r}: {lags_config}"
        )

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

        Parameters
        ----------
        models : StatsForecast model or list of models
            Statistical forecasting model(s) to fit on the trend component.
        values : ndarray of shape (n_samples,)
            Trend component values.
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
        self,
        group: pd.DataFrame,
        residual_part: np.ndarray,
        prediction_intervals=None,
        static_features: Optional[List[str]] = None,
    ) -> Any:
        """
        Fit an isolated copy of the base MLForecast pipeline on the extracted residual component.

        Parameters
        ----------
        group : pd.DataFrame
            Group-level source DataFrame containing time and exogenous features.
        residual_part : ndarray of shape (n_samples,)
            Residual values extracted from the trend decomposition.
        prediction_intervals : PredictionIntervals, optional
            Configuration for conformal prediction intervals.
        static_features : list of str, optional
            Exogenous columns whose values are constant within each series.

        Returns
        -------
        fitted_mf : MLForecast
            Fitted deep copy of the base MLForecast instance.
        """
        df_residual = group[[self.id_col_, self.time_col_] + self.exog_cols_].copy()
        df_residual[self.target_col_] = residual_part

        residual_callable = self._get_sku_config(
            self.residual_model_callable, group[self.id_col_].iloc[0]
        )
        if residual_callable is None:
            raise ValueError(
                f"'residual_model_callable' must be provided for unique_id "
                f"{group[self.id_col_].iloc[0]!r}."
            )
        if not callable(residual_callable):
            raise TypeError(
                f"residual_model_callable for unique_id "
                f"{group[self.id_col_].iloc[0]!r} must be callable."
            )

        computed_lags = self._get_residual_lags(
            group[self.id_col_].iloc[0], residual_part
        )
        try:
            mf_resid = residual_callable(nlags=computed_lags, freq=self.freq_)
        except TypeError:
            mf_resid = residual_callable(computed_lags, self.freq_)

        mf_resid.fit(
            df_residual,
            id_col=self.id_col_,
            time_col=self.time_col_,
            target_col=self.target_col_,
            prediction_intervals=prediction_intervals,
            static_features=static_features,
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
        static_features: Optional[List[str]] = None,
    ) -> "DTLWrapper":
        """
        Fit LOWESS trend decomposition and sub-models for each unique group.

        Parameters
        ----------
        df : pd.DataFrame
            Long-format panel dataset containing time series and exogenous
            features.
        id_col : str, default="unique_id"
            Column identifying individual time series.
        time_col : str, default="ds"
            Column identifying timestamps or integer time steps.
        target_col : str, default="y"
            Target variable column name.
        prediction_intervals : PredictionIntervals, optional
            MLForecast conformal interval configuration for residual uncertainty
            estimation.
        static_features : list of str, optional
            Exogenous columns that are constant within each series for the
            residual MLForecast model.

        Returns
        -------
        self : DTLWrapper
            Fitted estimator.

        Raises
        ------
        ValueError
            If ``freq`` or ``residual_model_callable`` is missing or invalid.
        """
        from statsforecast.models import AutoETS
        from tinyshift.series import detrend

        if self.freq is None:
            raise ValueError(
                "Parameter 'freq' must be explicitly declared when initializing DTLWrapper."
            )
        self.freq_ = self.freq

        self.id_col_ = id_col
        self.time_col_ = time_col
        self.target_col_ = target_col
        self.exog_cols_ = [
            c for c in df.columns if c not in [id_col, time_col, target_col]
        ]
        self.fitted_models_ = {}

        df_proc = df[[id_col, time_col, target_col] + self.exog_cols_].copy()

        if self.log_transform:
            df_proc[target_col] = np.log1p(df_proc[target_col])

        decomp_df = detrend(
            df_proc,
            frac=self.trend_frac,
            robust=self.robust,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )

        for uid, group in df.groupby(id_col):
            trend_model = self._get_trend_config(uid, partial(AutoETS, model="ZZN"))

            decomp_group = decomp_df.loc[group.index]
            dates = group[time_col]
            trend_part = decomp_group["trend"].to_numpy()
            residual_part = decomp_group["detrended"].to_numpy()

            sf_trend = self._fit_statsforecast(
                trend_model, trend_part, dates, uid, self.freq_
            )
            fitted_mf = self._fit_mlforecast(
                group,
                residual_part,
                prediction_intervals,
                static_features,
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

        Parameters
        ----------
        y_hat : ndarray of shape (h,)
            Raw recombined forecasts.
        method : {"hpi", "hfi"}
            Stabilization algorithm: 'hpi' (Horizontal Penalization Invariant)
            or 'hfi' (Horizontal Filtering Invariant).
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
        Generate future forecasts by recombining trend and residual predictions.

        Parameters
        ----------
        h : int
            Forecast horizon (number of steps ahead to predict).
        X_df : pd.DataFrame, optional
            DataFrame containing exogenous features for the forecast horizon.
        level : list of int or float, optional
            Confidence levels (0-100) for prediction intervals.
        stabilization_method : {"hpi", "hfi"}, optional
            Horizontal stabilization technique to post-process point and
            interval forecasts.
        w_s : float, default=0.0
            Stabilization weight parameter. Active only if `w_s > 0.0` and a
            method is selected.

        Returns
        -------
        preds_df : pd.DataFrame
            DataFrame with predictions from all MLForecast estimators, unique
            identifiers, timestamps, and optional prediction interval columns.

        Raises
        ------
        RuntimeError
            If the model is not fitted prior to calling predict.

        Notes
        -----
        - Recombination occurs in log space if `log_transform=True`, followed
          by scale inversion with `expm1`.
        - Trend projections are added to every model and interval column
          generated by MLForecast.
        """
        if not hasattr(self, "fitted_models_") or not self.fitted_models_:
            raise RuntimeError(
                "The model must be fitted with .fit() before making predictions."
            )
        if self.exog_cols_ and X_df is None:
            raise ValueError(
                f"The model was fitted with exogenous features {self.exog_cols_}. "
                "You must provide 'X_df' covering the forecast horizon."
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
