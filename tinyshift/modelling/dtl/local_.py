from typing import Any, List, Optional

import pandas as pd

from .base import BaseDTL


class DTLLocalWrapper(BaseDTL):
    """Fit one LOWESS residual MLForecast model per ``unique_id``."""

    def _fit_residuals(self, residuals, prediction_intervals, static_features) -> None:
        for uid, frame, lags in residuals:
            factory = self._get_sku_config(self.residual_model_callable, uid)
            if not callable(factory):
                raise ValueError(
                    f"residual_model_callable must be callable for unique_id {uid!r}."
                )
            try:
                model = factory(nlags=lags, freq=self.freq_)
            except TypeError:
                model = factory(lags, self.freq_)
            model.fit(
                frame,
                id_col=self.id_col_,
                time_col=self.time_col_,
                target_col=self.target_col_,
                prediction_intervals=prediction_intervals,
                static_features=static_features,
            )
            self.fitted_models_[uid]["residual"] = model

    def _predict_residuals(self, h, X_df, level):
        predictions = []
        for uid, models in self.fitted_models_.items():
            features = (
                X_df[X_df[self.id_col_] == uid].copy() if X_df is not None else None
            )
            predictions.append(
                models["residual"].predict(h=h, X_df=features, level=level)
            )
        return pd.concat(predictions, ignore_index=True)
