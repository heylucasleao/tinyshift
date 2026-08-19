# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from tinyshift.utils.imports import requires_extra

from .global_ import DMSTLGlobalWrapper
from .local_ import DMSTLLocalWrapper


class DMSTLWrapper(BaseEstimator, RegressorMixin):
    """Select the local or global DMSTL forecasting strategy.

    Decomposes each series in a panel into trend, seasonal, and residual
    components using MSTL. Trend and seasonal components are fitted with
    StatsForecast, while residual dynamics are fitted with MLForecast through
    the strategy selected by ``mode``. The facade is the public entry point;
    users do not need to import the strategy implementations directly.

    Parameters
    ----------
    mode : {"local", "global"}, default="global"
        Residual modeling strategy. ``"local"`` fits one MLForecast model per
        ``unique_id``. ``"global"`` fits one MLForecast model on the complete
        residual panel.
    residual_model_callable : callable or dict of callable
        Factory receiving ``nlags`` and ``freq`` and returning an MLForecast
        instance. In local mode, one factory may be supplied globally or a
        dictionary may map each ``unique_id`` to its own factory. In global
        mode, this must be one callable shared by the complete panel; a
        dictionary is not supported.
    freq : str or int, optional
        Frequency passed to StatsForecast and MLForecast.
    season_length : int, list of int, dict, or "auto", default="auto"
        Seasonal periods used by MSTL. ``"auto"`` detects candidate periods
        independently for each series. A dictionary can configure one integer,
        list of integers, or ``"auto"`` per ``unique_id``.
    seasonal_detection_params : dict, optional
        Keyword arguments passed to ``detect_seasonal_periods`` when
        ``season_length="auto"``.
    trend_model_callable : callable or dict of callable, optional
        Factory without arguments returning a StatsForecast trend model. A
        dictionary may provide one factory per ``unique_id``. If omitted,
        ``AutoETS(model="ZZN")`` is used.
    seasonal_model_callable : callable or dict of callable, optional
        Factory receiving one seasonal period and returning its StatsForecast
        model. A dictionary may provide one factory per ``unique_id``. If
        omitted, one ``SeasonalNaive`` model is created per period.
    nlags : int, list of int, dict, or "auto", default="auto"
        Residual lags. An integer expands to ``1..nlags`` and a list is used
        directly. With ``"auto"``, PAMI selects lags independently per series.
        In global mode, the resulting per-series lags are combined into their
        sorted union before the shared model is created.
    pami_params : dict, optional
        Keyword arguments passed to ``select_pami_lag`` when ``nlags="auto"``.
    log_transform : bool, default=False
        Apply ``log1p`` before decomposition and ``expm1`` after recombination.

    Attributes
    ----------
    delegate_ : DMSTLLocalWrapper or DMSTLGlobalWrapper
        Internal strategy selected by ``mode`` after fitting.
    fitted_models_ : dict
        Per-series trend and seasonal StatsForecast models. In local mode,
        each entry also contains its residual MLForecast model.
    residual_mlforecast_ : MLForecast
        Shared residual model available after fitting in global mode.

    Notes
    -----
    When ``season_length="auto"``, seasonal periods are detected independently
    for each series. When ``nlags="auto"``, PAMI is also evaluated per series.
    The global strategy then uses the union of the selected lags. Prediction
    recombines trend, seasonal, and residual forecasts and applies ``expm1``
    after recombination when ``log_transform=True``.

    Examples
    --------
    A residual factory receives the configured lags and frequency::

        from mlforecast import MLForecast
        from sklearn.ensemble import RandomForestRegressor

        def residual_model(nlags, freq):
            return MLForecast(
                models=[RandomForestRegressor(n_estimators=100, random_state=0)],
                lags=nlags,
                freq=freq,
            )

        model = DMSTLWrapper(
            mode="global",
            residual_model_callable=residual_model,
            freq="D",
            season_length="auto",
            seasonal_detection_params={"top_k": 2},
            nlags="auto",
            pami_params={"max_tau": 48, "m": 3, "delay": 1},
        )

        model.fit(df, id_col="unique_id", time_col="ds", target_col="y")
        predictions = model.predict(h=14)

    To use independent residual models, only the mode changes::

        model = DMSTLWrapper(
            mode="local",
            residual_model_callable=residual_model,
            freq="D",
            season_length={"series_a": [7, 30], "series_b": 7},
            nlags={"series_a": [1, 2, 3], "series_b": "auto"},
        )
    """

    def __init__(
        self,
        mode: Literal["local", "global"] = "global",
        residual_model_callable: Optional[Any] = None,
        freq: Optional[Union[str, int]] = None,
        season_length: Any = "auto",
        seasonal_detection_params: Optional[Dict[str, Any]] = None,
        trend_model_callable: Optional[Any] = None,
        seasonal_model_callable: Optional[Any] = None,
        nlags: Any = "auto",
        pami_params: Optional[Dict[str, Any]] = None,
        log_transform: bool = False,
    ) -> None:
        self.mode = mode
        self.residual_model_callable = residual_model_callable
        self.freq = freq
        self.season_length = season_length
        self.seasonal_detection_params = seasonal_detection_params
        self.trend_model_callable = trend_model_callable
        self.seasonal_model_callable = seasonal_model_callable
        self.nlags = nlags
        self.pami_params = pami_params
        self.log_transform = log_transform

    def _make_delegate(self):
        if self.mode == "local":
            strategy = DMSTLLocalWrapper
        elif self.mode == "global":
            strategy = DMSTLGlobalWrapper
        else:
            raise ValueError("mode must be either 'local' or 'global'.")

        return strategy(
            residual_model_callable=self.residual_model_callable,
            freq=self.freq,
            season_length=self.season_length,
            seasonal_detection_params=self.seasonal_detection_params,
            trend_model_callable=self.trend_model_callable,
            seasonal_model_callable=self.seasonal_model_callable,
            nlags=self.nlags,
            pami_params=self.pami_params,
            log_transform=self.log_transform,
        )

    @requires_extra("series")
    def fit(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        prediction_intervals: Optional[Any] = None,
        static_features: Optional[List[str]] = None,
    ) -> "DMSTLWrapper":
        """Fit the selected DMSTL strategy on a panel of time series.

        Each series is decomposed with MSTL. Trend and seasonal components
        are fitted with StatsForecast. Residual components are fitted either
        by one MLForecast model per ``unique_id`` when ``mode="local"`` or by
        one shared MLForecast model when ``mode="global"``.

        Parameters
        ----------
        df : pandas.DataFrame
            Long-format panel containing one row per observation. It must
            contain ``id_col``, ``time_col``, and ``target_col``. Any other
            columns are treated as exogenous features.
        id_col : str, default="unique_id"
            Column identifying each time series.
        time_col : str, default="ds"
            Column containing timestamps or ordered integer time steps.
        target_col : str, default="y"
            Column containing the observed target values.
        prediction_intervals : PredictionIntervals, optional
            Conformal prediction-interval configuration forwarded to the
            residual MLForecast model or models.
        static_features : list of str, optional
            Exogenous columns that remain constant within each series. These
            names are forwarded to MLForecast.

        Returns
        -------
        DMSTLWrapper
            The fitted wrapper. The selected implementation is available as
            ``delegate_`` and fitted attributes are copied to this object.

        Raises
        ------
        ValueError
            If ``mode`` is not ``"local"`` or ``"global"``, if ``freq`` is
            missing, or if seasonal periods or residual lags are invalid.
        ImportError
            If the optional ``series`` dependencies are not installed.
        """
        self.delegate_ = self._make_delegate()
        self.delegate_.fit(
            df,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            prediction_intervals=prediction_intervals,
            static_features=static_features,
        )
        for name, value in self.delegate_.__dict__.items():
            if name.endswith("_"):
                setattr(self, name, value)
        return self

    @requires_extra("series")
    def predict(
        self,
        h: int,
        X_df: Optional[pd.DataFrame] = None,
        level: Optional[List[Union[int, float]]] = None,
        stabilization_method: Optional[Literal["hpi", "hfi"]] = None,
        w_s: float = 0.0,
    ) -> pd.DataFrame:
        """Generate forecasts by recombining all DMSTL components.

        The selected strategy predicts residuals and the wrapper adds the
        trend and seasonal forecasts for each ``unique_id``. When
        ``log_transform=True``, recombination occurs in log space and the
        result is transformed back with ``expm1`` before being returned.

        Parameters
        ----------
        h : int
            Number of future time steps to forecast.
        X_df : pandas.DataFrame, optional
            Future exogenous features. It must contain the identifier column
            and rows for the forecast horizon of every series when exogenous
            columns were present during fitting. In local mode, rows are
            filtered per ``unique_id`` before prediction; in global mode, the
            complete frame is passed to the shared residual model.
        level : list of int or float, optional
            Confidence levels between 0 and 100 for prediction intervals,
            such as ``[80, 95]``. Forwarded to MLForecast.
        stabilization_method : {"hpi", "hfi"}, optional
            Optional horizontal stabilization applied to point and interval
            forecasts after recombination.
        w_s : float, default=0.0
            Stabilization weight. Stabilization is applied only when a method
            is selected and ``w_s > 0``.

        Returns
        -------
        pandas.DataFrame
            Forecasts containing the identifier and time columns returned by
            MLForecast, together with model prediction and optional interval
            columns.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If exogenous features were used during fitting but ``X_df`` is
            missing, or if the stabilization method is invalid.
        ImportError
            If the optional ``series`` dependencies are not installed.
        """
        if not hasattr(self, "delegate_"):
            raise RuntimeError(
                "The model must be fitted with .fit() before making predictions."
            )
        return self.delegate_.predict(
            h=h,
            X_df=X_df,
            level=level,
            stabilization_method=stabilization_method,
            w_s=w_s,
        )
