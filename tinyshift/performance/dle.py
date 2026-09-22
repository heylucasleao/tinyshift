# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

"""Direct estimation of squared loss for regression or binary probabilities."""

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_array, check_is_fitted


class DirectLossEstimator(BaseEstimator):
    """Learn per-row squared error from labeled reference data.

    For regression, this estimates MSE. For binary classification, pass the
    probability of class 1 as ``y_pred`` and labels encoded as 0 and 1; the
    resulting MSE is the binary Brier score. The loss model receives numeric
    features and the monitored model's prediction as its last feature.

    Parameters
    ----------
    estimator : sklearn regressor or None, default=None
        Template for the loss model, cloned during fitting. The default is a
        random forest regressor.
    """

    def __init__(self, estimator=None) -> None:
        self.estimator = estimator

    @staticmethod
    def _inputs(X, y_pred):
        features = check_array(X, ensure_2d=True, dtype=float)
        predictions = DirectLossEstimator._vector(y_pred, "y_pred")
        if len(features) != len(predictions):
            raise ValueError("X and y_pred must have the same number of rows.")
        return features, predictions

    @staticmethod
    def _target(y_true, n_samples):
        target = DirectLossEstimator._vector(y_true, "y_true")
        if len(target) != n_samples:
            raise ValueError("X and y_true must have the same number of rows.")
        return target

    @staticmethod
    def _vector(values, name):
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        return check_array(array.reshape(-1, 1), dtype=float).ravel()

    def observed_loss(self, y_true, y_pred):
        """Return squared error for each observation."""
        predictions = self._vector(y_pred, "y_pred")
        target = self._target(y_true, len(predictions))
        return (target - predictions) ** 2

    def aggregate(self, losses) -> float:
        """Return the mean squared error (binary Brier score for probabilities)."""
        values = self._vector(losses, "losses")
        if np.any(values < 0):
            raise ValueError("losses must be nonnegative.")
        return float(np.mean(values))

    def fit(self, X, y_true, y_pred):
        """Fit the loss model on labeled reference observations."""
        features, predictions = self._inputs(X, y_pred)
        losses = self.observed_loss(y_true, predictions)
        template = self.estimator
        if template is None:
            template = RandomForestRegressor(
                n_estimators=100, min_samples_leaf=3, random_state=42
            )
        self.loss_model_ = clone(template)
        self.loss_model_.fit(np.column_stack((features, predictions)), losses)
        self.n_features_in_ = features.shape[1]
        return self

    def estimate_loss(self, X, y_pred) -> np.ndarray:
        """Predict nonnegative per-row squared losses without current targets."""
        check_is_fitted(self, "loss_model_")
        features, predictions = self._inputs(X, y_pred)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {features.shape[1]} features, expected {self.n_features_in_}."
            )
        losses = np.asarray(
            self.loss_model_.predict(np.column_stack((features, predictions))),
            dtype=float,
        )
        if losses.shape != (len(features),) or not np.isfinite(losses).all():
            raise ValueError("The loss model must return one finite loss per row.")
        return np.maximum(losses, 0.0)

    def estimate(self, X, y_pred) -> float:
        """Estimate MSE or binary Brier score for an unlabeled batch."""
        return self.aggregate(self.estimate_loss(X, y_pred))
