import copy
import pandas as pd
import numpy as np
from typing import Literal
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.seasonal import MSTL, DecomposeResult
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive
from mlforecast import MLForecast
from tinyshift.series import hpi, hfi


class DMSTLWrapper(BaseEstimator, RegressorMixin):
    """
    Decomposed Multiple Seasonal-Trend (DMSTL) local wrapper per unique_id.

    Decomposes multi-seasonal time series into trend, seasonal, and residual
    components using MSTL. Fits statistical base models on trend and seasonal
    components via StatsForecast, while modelling complex non-linear residual
    dynamics using MLForecast. Optionally applies log-additive transformations
    (Box-Cox/Log-1p) and horizontal stabilization (HPI/HFI).

    Parameters
    ----------
    mf_resid : MLForecast
        Base MLForecast pipeline configured with machine learning estimators
        to fit the residual component.
    season_length : int or list of int
        Seasonal period(s) passed directly to MSTL decomposition.
    freq : str or int, optional
        Frequency of the time series (e.g., 'D', 'H', 1). If None, attempts
        to infer from `mf_resid.freq`.
    trend_model : StatsForecast model, optional
        Statistical model instance for the trend component. Defaults to
        AutoETS(model="MMN") if None.
    seasonal_model : StatsForecast model or list of models, optional
        Statistical model(s) for the seasonal component(s). Defaults to
        SeasonalNaive for each period in `season_length`.
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
        Effective time series frequency.
    trend_model_ : StatsForecast model
        Configured trend model.
    seasonal_model_ : list of StatsForecast models
        Configured seasonal models.
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

    def __init__(
        self,
        mf_resid: MLForecast,
        season_length: int | list[int],
        freq: str | int | None = None,
        trend_model=None,
        seasonal_model=None,
        log_transform: bool = False,
    ):
        self.mf_resid = mf_resid
        self.season_length = season_length
        self.freq = freq
        self.trend_model = trend_model
        self.seasonal_model = seasonal_model
        self.log_transform = log_transform

    def _get_model_cols(self, df: pd.DataFrame) -> list[str]:
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

    def _extract_components_df(self, result: DecomposeResult) -> pd.DataFrame:
        """
        Extract decomposed components from a statsmodels MSTL DecomposeResult.

        Parameters
        ----------
        result : DecomposeResult
            Fitted MSTL decomposition result object.

        Returns
        -------
        components_df : pd.DataFrame
            Structured DataFrame containing original data, trend, seasonal,
            and residual components.
        """
        df = pd.DataFrame()
        df["data"] = result.observed
        df["trend"] = result.trend

        seasonal = np.asarray(result.seasonal)
        if seasonal.ndim == 1:
            df["seasonal"] = seasonal
        else:
            for i in range(seasonal.shape[1]):
                df[f"seasonal_{i}"] = seasonal[:, i]

        df["resid"] = result.resid
        return df

    def _process_components(
        self, components_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        seasonal_part : ndarray of shape (n_samples,)
            Summed seasonal component across all seasonal channels.
        residual_part : ndarray of shape (n_samples,)
            Zero-filled residual sequence.
        """
        trend_part = components_df["trend"].bfill().ffill().values
        seasonal_cols = [c for c in components_df.columns if c.startswith("seasonal")]
        seasonal_part = components_df[seasonal_cols].sum(axis=1).values
        residual_part = components_df["resid"].fillna(0.0).values
        return trend_part, seasonal_part, residual_part

    def _fit_statsforecast(
        self, models, values: np.ndarray, dates: pd.Series, uid, freq
    ) -> StatsForecast:
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
        sf_df = pd.DataFrame(
            {self.id_col_: uid, self.time_col_: dates, self.target_col_: values}
        )
        models_list = models if isinstance(models, list) else [models]
        return StatsForecast(models=models_list, freq=freq).fit(sf_df)

    def _fit_mlforecast(
        self, group: pd.DataFrame, residual_part: np.ndarray, prediction_intervals=None
    ) -> MLForecast:
        """
        Fit an MLForecast pipeline on the extracted residual component.

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
        mf_resid : MLForecast
            Fitted deep copy of the MLForecast instance.
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

    def fit(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        prediction_intervals=None,
    ):
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
            If frequency is not specified in init or inferred from `mf_resid`.
        """
        self.season_length_ = (
            [self.season_length]
            if isinstance(self.season_length, int)
            else self.season_length
        )
        self.freq_ = (
            self.freq if self.freq is not None else getattr(self.mf_resid, "freq", None)
        )

        if self.freq_ is None:
            raise ValueError(
                "Parameter 'freq' must be explicitly provided in DMSTLWrapper "
                "or pre-defined in the 'mf_resid' instance."
            )

        self.trend_model_ = (
            self.trend_model if self.trend_model is not None else AutoETS(model="MMN")
        )

        if self.seasonal_model is not None:
            self.seasonal_model_ = (
                self.seasonal_model
                if isinstance(self.seasonal_model, list)
                else [self.seasonal_model]
            )
        else:
            self.seasonal_model_ = [
                SeasonalNaive(season_length=sl) for sl in self.season_length_
            ]

        self.id_col_ = id_col
        self.time_col_ = time_col
        self.target_col_ = target_col
        self.exog_cols_ = [
            c for c in df.columns if c not in [id_col, time_col, target_col]
        ]
        self.fitted_models_ = {}

        for uid, group in df.groupby(id_col):
            group_sorted = group.sort_values(time_col).copy()
            y_series = group_sorted[target_col].values

            if self.log_transform:
                y_series = np.log1p(y_series)

            dates = group_sorted[time_col]

            mstl = MSTL(y_series, periods=self.season_length_)
            res = mstl.fit()
            components_df = self._extract_components_df(res)
            trend_part, seasonal_part, residual_part = self._process_components(
                components_df
            )

            sf_trend = self._fit_statsforecast(
                self.trend_model_, trend_part, dates, uid, self.freq_
            )
            sf_seasonal = self._fit_statsforecast(
                self.seasonal_model_, seasonal_part, dates, uid, self.freq_
            )
            mf_resid = self._fit_mlforecast(
                group_sorted, residual_part, prediction_intervals
            )

            self.fitted_models_[uid] = {
                "trend": sf_trend,
                "seasonal": sf_seasonal,
                "residual": mf_resid,
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
        if method == "hpi":
            return hpi(y_hat, w_s=w_s)
        elif method == "hfi":
            return hfi(y_hat, w_s=w_s)
        else:
            raise ValueError(
                f"Invalid method '{method}'. Choose either 'hpi' or 'hfi'."
            )

    def predict(
        self,
        h: int,
        X_df: pd.DataFrame | None = None,
        level: list[int | float] | None = None,
        stabilization_method: Literal["hpi", "hfi"] | None = None,
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
            DataFrame with predictions, unique IDs, timestamps, and optional prediction interval columns.

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
            df_seasonal = sf_seasonal.predict(h=h)

            trend_cols = self._get_model_cols(df_trend)
            seasonal_cols = self._get_model_cols(df_seasonal)

            trend_preds = df_trend[trend_cols].sum(axis=1).values
            seasonal_preds = df_seasonal[seasonal_cols].sum(axis=1).values

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
