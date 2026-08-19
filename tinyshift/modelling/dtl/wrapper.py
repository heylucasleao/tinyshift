# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from tinyshift.utils.imports import requires_extra

from .global_ import DTLGlobalWrapper
from .local_ import DTLLocalWrapper


class DTLWrapper(BaseEstimator, RegressorMixin):
    """Select the local or global LOWESS-based DTL forecasting strategy.

    DTL decomposes each non-seasonal series into a robust LOWESS trend and a
    residual component. Trend forecasts are fitted separately per
    ``unique_id`` in both modes. The residual strategy is selected with
    ``mode``: ``"local"`` fits one residual MLForecast model per series,
    while ``"global"`` fits one residual model on the complete residual
    panel. The facade is the public entry point; the strategy implementations
    do not need to be imported directly.

    Parameters
    ----------
    mode : {"local", "global"}, default="global"
        Residual modeling strategy.
    residual_model_callable : callable or dict of callable, optional
        Factory receiving ``nlags`` and ``freq`` and returning an MLForecast
        instance. Local mode accepts one global factory or a dictionary mapping
        each ``unique_id`` to its own factory. Global mode requires one shared
        callable; a dictionary is not supported.
    freq : str or int, optional
        Frequency passed to StatsForecast and MLForecast.
    trend_model_callable : callable or dict of callable, optional
        Factory without arguments returning a StatsForecast trend model.
    trend_frac : float, default=0.2
        Fraction used by LOWESS to estimate each trend value.
    robust : bool, default=True
        Whether LOWESS applies robustifying iterations.
    log_transform : bool, default=False
        Apply ``log1p`` before decomposition and ``expm1`` after recombination.
    nlags : int, list of int, dict, or "auto", default="auto"
        Residual lags. An integer expands to ``1..nlags`` and a list is used
        directly. With ``"auto"``, PAMI selects lags independently per series.
        In global mode, the resulting lags are combined into their sorted union
        before the shared residual model is created.
    pami_params : dict, optional
        Keyword arguments passed to ``select_pami_lag`` when ``nlags="auto"``.

    Attributes
    ----------
    delegate_ : DTLLocalWrapper or DTLGlobalWrapper
        Internal strategy selected by ``mode`` after fitting.
    fitted_models_ : dict
        Trend StatsForecast models mapped by ``unique_id``. In local mode,
        each entry also contains its residual MLForecast model.
    residual_mlforecast_ : MLForecast
        Shared residual model available after fitting in global mode.

    Notes
    -----
    ``DTLWrapper`` is intended for non-seasonal data. The LOWESS trend and its
    StatsForecast model are always fitted independently for each series. In
    local mode, ``X_df`` is filtered by ``unique_id`` and passed to each
    residual model separately. In global mode, the complete future feature
    frame is passed to the shared residual model. When ``nlags="auto"``, PAMI
    is evaluated per series in both modes; global mode then uses the union of
    the selected lags.

    Examples
    --------
    A shared residual factory can be used in either mode::

        from mlforecast import MLForecast
        from sklearn.ensemble import RandomForestRegressor

        def residual_model(nlags, freq):
            return MLForecast(
                models=[RandomForestRegressor(n_estimators=100, random_state=0)],
                lags=nlags,
                freq=freq,
            )

        model = DTLWrapper(
            mode="global",
            residual_model_callable=residual_model,
            freq="D",
            trend_frac=0.2,
            robust=True,
            nlags="auto",
            pami_params={"max_tau": 48, "m": 3, "delay": 1},
        )

        model.fit(df, id_col="unique_id", time_col="ds", target_col="y")
        predictions = model.predict(h=14)

    Local residual models may be configured per series::

        model = DTLWrapper(
            mode="local",
            residual_model_callable={
                "series_a": residual_model,
                "series_b": residual_model,
            },
            freq="D",
            nlags={"series_a": [1, 2, 3], "series_b": "auto"},
        )
    """

    def __init__(
        self,
        mode: Literal["local", "global"] = "global",
        residual_model_callable: Optional[Any] = None,
        freq: Optional[Union[str, int]] = None,
        trend_model_callable: Optional[Any] = None,
        trend_frac: float = 0.2,
        robust: bool = True,
        log_transform: bool = False,
        nlags: Any = "auto",
        pami_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.mode = mode
        self.residual_model_callable = residual_model_callable
        self.freq = freq
        self.trend_model_callable = trend_model_callable
        self.trend_frac = trend_frac
        self.robust = robust
        self.log_transform = log_transform
        self.nlags = nlags
        self.pami_params = pami_params

    def _make_delegate(self):
        if self.mode == "local":
            strategy = DTLLocalWrapper
        elif self.mode == "global":
            strategy = DTLGlobalWrapper
        else:
            raise ValueError("mode must be either 'local' or 'global'.")
        return strategy(
            residual_model_callable=self.residual_model_callable,
            freq=self.freq,
            trend_model_callable=self.trend_model_callable,
            trend_frac=self.trend_frac,
            robust=self.robust,
            log_transform=self.log_transform,
            nlags=self.nlags,
            pami_params=self.pami_params,
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
    ) -> "DTLWrapper":
        """Fit the selected LOWESS trend and residual strategy.

        Each series is decomposed into a LOWESS trend and residual component.
        The trend is fitted with StatsForecast independently for every
        ``unique_id``. Residuals are fitted by one MLForecast model per series
        in local mode or by one shared MLForecast model in global mode.

        Parameters
        ----------
        df : pandas.DataFrame
            Long-format panel containing ``id_col``, ``time_col``, and
            ``target_col``. Other columns are treated as exogenous features.
        id_col : str, default="unique_id"
            Column identifying each time series.
        time_col : str, default="ds"
            Column containing timestamps or ordered integer time steps.
        target_col : str, default="y"
            Column containing observed target values.
        prediction_intervals : PredictionIntervals, optional
            Conformal interval configuration forwarded to the residual
            MLForecast model or models.
        static_features : list of str, optional
            Exogenous columns constant within each series. Forwarded to
            MLForecast.

        Returns
        -------
        DTLWrapper
            The fitted wrapper. The selected implementation is available as
            ``delegate_`` and its fitted attributes are copied to this object.

        Raises
        ------
        ValueError
            If ``mode`` is invalid, ``freq`` is missing, or residual lags are
            invalid.
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
        """Predict by recombining LOWESS trend and residual forecasts.

        The selected strategy forecasts residuals and this facade adds the
        trend forecast for each ``unique_id``. When ``log_transform=True``,
        recombination occurs in log space and values are transformed back with
        ``expm1`` before being returned.

        Parameters
        ----------
        h : int
            Number of future time steps to forecast.
        X_df : pandas.DataFrame, optional
            Future exogenous features. It must contain the identifier column
            and rows for the forecast horizon of every series when exogenous
            columns were present during fitting. Local mode filters this frame
            by ``unique_id`` before each residual prediction; global mode
            passes the complete frame to the shared residual model.
        level : list of int or float, optional
            Confidence levels between 0 and 100, such as ``[80, 95]``.
            Forwarded to MLForecast.
        stabilization_method : {"hpi", "hfi"}, optional
            Horizontal stabilization applied to point and interval forecasts
            after trend and residual recombination.
        w_s : float, default=0.0
            Stabilization weight. Applied only when a method is selected and
            ``w_s > 0``.

        Returns
        -------
        pandas.DataFrame
            Forecasts containing identifier and time columns, model
            predictions, and optional prediction-interval columns.

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
