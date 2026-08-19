# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Optional, List, Any

import pandas as pd

from .base import BaseDTL


class DTLGlobalWrapper(BaseDTL):
    """Fit one LOWESS residual MLForecast model for the complete panel."""

    def _fit_residuals(self, residuals, prediction_intervals, static_features) -> None:
        factory = self.residual_model_callable
        if not callable(factory):
            raise ValueError(
                "residual_model_callable must be one callable in global mode."
            )
        lags = sorted(
            {lag for _, _, series_lags in residuals for lag in series_lags}
        ) or [1]
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

    def _predict_residuals(self, h, X_df, level):
        return self.residual_mlforecast_.predict(h=h, X_df=X_df, level=level)
