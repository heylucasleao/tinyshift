# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

"""Panel adapter for direct loss estimation."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from .confidence import ConfidenceBasedPerformanceEstimator
from .dle import DirectLossEstimator


class DirectLossAnalyzer(BaseEstimator):
    """Estimate regression performance independently for each panel ID.

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
        self.estimator = estimator
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
        template = DirectLossEstimator() if self.estimator is None else self.estimator
        if not isinstance(template, DirectLossEstimator):
            raise TypeError("estimator must be a DirectLossEstimator.")

        estimators = {}
        baselines = {}
        for unique_id, group in reference.groupby(id_col, sort=False, observed=True):
            split = int(np.floor(len(group) * (1 - self.validation_fraction)))
            if split < 2 or split == len(group):
                raise ValueError(
                    f"ID {unique_id!r} needs at least two fitting rows and one held-out row."
                )
            train, holdout = group.iloc[:split], group.iloc[split:]
            fitted = clone(template).fit(
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

    def predict(self, current: pd.DataFrame) -> pd.DataFrame:
        """Return estimated metric and change for each current ID."""
        check_is_fitted(self, "estimators_")
        self._validate_frame(
            current, [self.id_col_, self.prediction_col_, *self.feature_cols_], self.id_col_
        )
        unknown = [
            value for value in pd.unique(current[self.id_col_])
            if value not in self.estimators_
        ]
        if unknown:
            raise ValueError(f"No reference performance for IDs: {unknown!r}.")

        rows = []
        for unique_id, group in current.groupby(
            self.id_col_, sort=False, observed=True
        ):
            baseline = self.baselines_[unique_id]
            estimated = self.estimators_[unique_id].estimate(
                group[self.feature_cols_], group[self.prediction_col_]
            )
            delta = estimated - baseline["reference_estimated"]
            rows.append(
                {
                    self.id_col_: unique_id,
                    "metric": self.estimators_[unique_id].metric,
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


class ConfidenceBasedPerformanceAnalyzer(BaseEstimator):
    """Estimate classification performance for each panel ID.

    ``probability_cols`` maps each class label to its probability column. It
    must include every class in the same order in reference and current data.
    The reference must contain observed labels; the current batch need not.
    Unlike regression DLE, this estimator does not train a loss model, so the
    complete reference is used for its baseline.
    """

    def __init__(self, estimator: ConfidenceBasedPerformanceEstimator | None = None):
        self.estimator = estimator

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
        probability_cols: dict,
        id_col: str = "unique_id",
        target_col: str = "y",
    ) -> "ConfidenceBasedPerformanceAnalyzer":
        """Fit one probability-based estimator per reference ID."""
        if not isinstance(probability_cols, dict) or len(probability_cols) < 2:
            raise ValueError("probability_cols must map at least two classes to columns.")
        columns = list(probability_cols.values())
        if len(set(columns)) != len(columns) or set(columns) & {id_col, target_col}:
            raise ValueError("Probability columns must be distinct from ID and target.")
        self._validate_frame(reference, [id_col, target_col, *columns], id_col)
        template = (
            ConfidenceBasedPerformanceEstimator()
            if self.estimator is None
            else self.estimator
        )
        if not isinstance(template, ConfidenceBasedPerformanceEstimator):
            raise TypeError("estimator must be a ConfidenceBasedPerformanceEstimator.")

        estimators = {}
        for unique_id, group in reference.groupby(id_col, sort=False, observed=True):
            estimators[unique_id] = clone(template).fit(
                group[target_col].to_numpy(), group[columns], list(probability_cols)
            )
        self.estimators_ = estimators
        self.probability_cols_ = dict(probability_cols)
        self.id_col_ = id_col
        self.target_col_ = target_col
        return self

    def predict(self, current: pd.DataFrame) -> pd.DataFrame:
        """Return estimated performance and change for each current ID."""
        check_is_fitted(self, "estimators_")
        columns = list(self.probability_cols_.values())
        self._validate_frame(current, [self.id_col_, *columns], self.id_col_)
        unknown = [
            value for value in pd.unique(current[self.id_col_])
            if value not in self.estimators_
        ]
        if unknown:
            raise ValueError(f"No reference performance for IDs: {unknown!r}.")

        rows = []
        for unique_id, group in current.groupby(
            self.id_col_, sort=False, observed=True
        ):
            fitted = self.estimators_[unique_id]
            estimated = fitted.estimate(group[columns])
            delta = estimated - fitted.reference_estimated_
            rows.append(
                {
                    self.id_col_: unique_id,
                    "metric": fitted.metric,
                    "reference_realized": fitted.reference_realized_,
                    "reference_estimated": fitted.reference_estimated_,
                    "reference_size": fitted.reference_size_,
                    "current_estimated": estimated,
                    "estimated_delta": delta,
                    "degradation": bool(delta < 0),
                    "current_size": len(group),
                }
            )
        self.results_ = pd.DataFrame(rows)
        return self.results_.copy()

    def summary(self) -> pd.DataFrame:
        """Return a copy of the most recent prediction result."""
        check_is_fitted(self, "results_")
        return self.results_.copy()
