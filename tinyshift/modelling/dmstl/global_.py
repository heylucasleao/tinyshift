# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Any, List, Optional, Union

import pandas as pd

from .base import BaseDMSTL


class DMSTLGlobalWrapper(BaseDMSTL):
    """MSTL forecaster with one residual MLForecast model for the panel.

    Trend and each seasonal component are still decomposed and fitted
    independently for every ``unique_id``. In contrast with
    :class:`DMSTLLocalWrapper`, all residual components are concatenated into
    one panel and fitted by a single MLForecast instance. The global residual
    factory therefore receives the union of the configured lags across all
    series and must be one callable shared by the complete panel.

    Parameters
    ----------
    residual_model_callable : callable
        Factory receiving ``nlags`` and ``freq`` and returning the MLForecast
        instance used for all residual series. A dictionary by ``unique_id`` is
        not supported in global mode.
    freq : str or int
        Frequency passed to StatsForecast and MLForecast.
    season_length : int, list of int, dict, or "auto", default="auto"
        Seasonal periods used by MSTL. A dictionary may configure periods per
        ``unique_id``; ``"auto"`` detects them from each series.
    nlags : int, list of int, dict, or "auto", default="auto"
        Residual lags. PAMI is evaluated per series when set to ``"auto"``;
        the global model receives the sorted union of all resulting lags.
    log_transform : bool, default=False
        Apply ``log1p`` before decomposition and ``expm1`` after recombination.

    Notes
    -----
    Prediction calls the shared residual model once for the complete panel.
    The resulting rows are then matched by ``unique_id`` and recombined with
    that series' trend and seasonal forecasts. ``X_df`` must contain future
    features for every series when exogenous columns were used during fitting.
    """

    def _fit_residuals(
        self,
        residuals,
        prediction_intervals: Optional[Any],
        static_features: Optional[List[str]],
    ) -> None:
        factory = self.residual_model_callable
        if not callable(factory):
            raise ValueError(
                "residual_model_callable must be one callable in global mode."
            )
        # PAMI is evaluated above per SKU; the panel model receives their union.
        lags = sorted({lag for _, _, sku_lags in residuals for lag in sku_lags}) or [1]
        try:
            self.residual_mlforecast_ = factory(nlags=lags, freq=self.freq_)
        except TypeError:
            self.residual_mlforecast_ = factory(lags, self.freq_)
        panel = pd.concat([frame for _, frame, _ in residuals], ignore_index=True)
        self.residual_mlforecast_.fit(
            panel,
            id_col=self.id_col_,
            time_col=self.time_col_,
            target_col=self.target_col_,
            prediction_intervals=prediction_intervals,
            static_features=static_features,
        )

    def _predict_residuals(
        self,
        h: int,
        X_df: Optional[pd.DataFrame],
        level: Optional[List[Union[int, float]]],
    ) -> pd.DataFrame:
        return self.residual_mlforecast_.predict(h=h, X_df=X_df, level=level)
