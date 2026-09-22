# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

"""Panel adapter for direct loss estimation."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from .dle import DirectLossEstimator


class DirectLossAnalyzer(BaseEstimator):
    """Estimate squared loss independently for each panel ID.

    The first ``1 - validation_fraction`` of each reference group fits its loss
    model. The remaining observations form a held-out baseline. Order rows
    chronologically within each ID when using the split for time series.

    ``degradation`` indicates a positive change in estimated metric, not a
    statistical hypothesis test or a verified change in realized performance.
    """

    def __init__(
        self,
        estimator: DirectLossEstimator | None = None,
        validation_fraction: float = 0.25,
    ) -> None:
        if estimator is not None and not isinstance(estimator, DirectLossEstimator):
            raise TypeError("estimator must be a DirectLossEstimator.")
        self.estimator = DirectLossEstimator() if estimator is None else estimator
        self.validation_fraction = validation_fraction

    @staticmethod
    def _validate_frame(df, required, id_col):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Panel input cannot be empty.")
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}.")
        if df[id_col].isna().any():
            raise ValueError("ID values must not be missing.")

    def fit(
        self,
        reference: pd.DataFrame,
        feature_cols: list[str],
        id_col: str = "unique_id",
        target_col: str = "y",
        prediction_col: str = "y_pred",
    ) -> "DirectLossAnalyzer":
        """Fit one loss estimator per ID on labeled reference predictions."""
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must lie strictly between 0 and 1.")
        if not feature_cols or len(feature_cols) != len(set(feature_cols)):
            raise ValueError("feature_cols must contain distinct feature names.")
        if set(feature_cols) & {id_col, target_col, prediction_col}:
            raise ValueError("feature_cols must exclude ID, target, and prediction columns.")
        self._validate_frame(
            reference, [id_col, target_col, prediction_col, *feature_cols], id_col
        )
        estimators = {}
        baselines = {}
        for unique_id, group in reference.groupby(id_col, sort=False, observed=True):
            split = int(np.floor(len(group) * (1 - self.validation_fraction)))
            if split < 2 or split == len(group):
                raise ValueError(
                    f"ID {unique_id!r} needs at least two fitting rows and one held-out row."
                )
            train, holdout = group.iloc[:split], group.iloc[split:]
            fitted = clone(self.estimator).fit(
                train[feature_cols], train[target_col], train[prediction_col]
            )
            baselines[unique_id] = {
                "reference_estimated": fitted.estimate(
                    holdout[feature_cols], holdout[prediction_col]
                ),
                "reference_realized": fitted.aggregate(
                    fitted.observed_loss(holdout[target_col], holdout[prediction_col])
                ),
                "reference_size": len(holdout),
            }
            estimators[unique_id] = fitted

        self.estimators_ = estimators
        self.baselines_ = baselines
        self.feature_cols_ = list(feature_cols)
        self.id_col_ = id_col
        self.target_col_ = target_col
        self.prediction_col_ = prediction_col
        return self

    def predict(
        self,
        current: pd.DataFrame,
        id_col: str | None = None,
        prediction_col: str | None = None,
    ) -> pd.DataFrame:
        """Return estimated loss and change for each current ID.

        Optional column names let the current frame use different ID and
        prediction columns from the reference frame. Feature names are shared.
        """
        check_is_fitted(self, "estimators_")
        id_col = self.id_col_ if id_col is None else id_col
        prediction_col = self.prediction_col_ if prediction_col is None else prediction_col
        self._validate_frame(
            current, [id_col, prediction_col, *self.feature_cols_], id_col
        )
        unknown = [
            value for value in pd.unique(current[id_col])
            if value not in self.estimators_
        ]
        if unknown:
            raise ValueError(f"No reference performance for IDs: {unknown!r}.")

        rows = []
        for unique_id, group in current.groupby(
            id_col, sort=False, observed=True
        ):
            baseline = self.baselines_[unique_id]
            estimated = self.estimators_[unique_id].estimate(
                group[self.feature_cols_], group[prediction_col]
            )
            delta = estimated - baseline["reference_estimated"]
            rows.append(
                {
                    id_col: unique_id,
                    "metric": "mse",
                    **baseline,
                    "current_estimated": estimated,
                    "estimated_delta": delta,
                    "degradation": bool(delta > 0),
                    "current_size": len(group),
                }
            )
        self.results_ = pd.DataFrame(rows)
        return self.results_.copy()

    def summary(self) -> pd.DataFrame:
        """Return a copy of the most recent prediction result."""
        check_is_fitted(self, "results_")
        return self.results_.copy()
