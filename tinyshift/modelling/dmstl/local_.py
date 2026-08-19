from typing import Any, List, Optional, Union

import pandas as pd

from .base import BaseDMSTL


class DMSTLLocalWrapper(BaseDMSTL):
    """MSTL forecaster with one residual MLForecast model per series.

    Trend and each seasonal component are fitted independently with
    StatsForecast for every ``unique_id``. The residual component is fitted
    with a separate MLForecast instance for each series, allowing
    ``residual_model_callable`` and ``nlags`` to be configured globally or by
    ``unique_id``.

    Parameters
    ----------
    residual_model_callable : callable or dict of callable
        Factory receiving ``nlags`` and ``freq`` and returning a fitted-model
        compatible MLForecast instance. A dictionary may provide one factory
        per ``unique_id``.
    freq : str or int
        Frequency passed to StatsForecast and MLForecast.
    season_length : int, list of int, dict, or "auto", default="auto"
        Seasonal periods used by MSTL. A dictionary may configure periods per
        ``unique_id``; ``"auto"`` detects them from each series.
    nlags : int, list of int, dict, or "auto", default="auto"
        Residual lags. An integer expands to ``1..nlags`` and a dictionary may
        configure them per ``unique_id``.
    log_transform : bool, default=False
        Apply ``log1p`` before decomposition and ``expm1`` after recombination.

    Notes
    -----
    During prediction, ``X_df`` is filtered by ``unique_id`` before being sent
    to each residual model. This strategy is appropriate when series have
    distinct residual dynamics or require distinct residual model factories.
    """

    def _fit_residuals(
        self,
        residuals,
        prediction_intervals: Optional[Any],
        static_features: Optional[List[str]],
    ) -> None:
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

    def _predict_residuals(
        self,
        h: int,
        X_df: Optional[pd.DataFrame],
        level: Optional[List[Union[int, float]]],
    ) -> pd.DataFrame:
        predictions = []
        for uid, models in self.fitted_models_.items():
            features = (
                X_df[X_df[self.id_col_] == uid].copy() if X_df is not None else None
            )
            predictions.append(
                models["residual"].predict(h=h, X_df=features, level=level)
            )
        return pd.concat(predictions, ignore_index=True)
