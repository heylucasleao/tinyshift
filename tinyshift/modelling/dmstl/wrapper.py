from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from tinyshift.utils.imports import requires_extra

from .global_ import DMSTLGlobalWrapper
from .local_ import DMSTLLocalWrapper


class DMSTLWrapper(BaseEstimator, RegressorMixin):
    """Select the local or global DMSTL forecasting strategy.

    Parameters
    ----------
    mode : {"local", "global"}, default="global"
        Residual modeling strategy. ``"local"`` fits one MLForecast model per
        ``unique_id``; ``"global"`` fits one MLForecast model on the complete
        residual panel.
    residual_model_callable : callable or dict of callable
        Factory for the residual MLForecast model. Local mode accepts one
        factory or a factory per ``unique_id``. Global mode requires one
        callable shared by the complete panel.
    freq : str or int
        Frequency passed to StatsForecast and MLForecast.
    season_length : int, list of int, dict, or "auto", default="auto"
        Seasonal periods used by MSTL. A dictionary can configure periods per
        ``unique_id``.
    seasonal_detection_params : dict, optional
        Parameters passed to automatic seasonal-period detection.
    trend_model_callable : callable or dict of callable, optional
        Factory for each series' trend model. Local and global modes support
        per-series factories.
    seasonal_model_callable : callable or dict of callable, optional
        Factory receiving a seasonal period and returning its model.
    nlags : int, list of int, dict, or "auto", default="auto"
        Residual lags. In global mode, per-series lags are combined into their
        sorted union before the shared model is created.
    pami_params : dict, optional
        Parameters passed to automatic PAMI lag selection.
    log_transform : bool, default=False
        Apply ``log1p`` before decomposition and ``expm1`` after recombination.

    Notes
    -----
    The selected strategy is created on the first call to :meth:`fit` and is
    available as ``delegate_`` after fitting. Fitted attributes such as
    ``fitted_models_`` and ``residual_mlforecast_`` are exposed on this facade.
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
        """Fit the selected local or global DMSTL strategy."""
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
        """Predict with the strategy selected during initialization."""
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
