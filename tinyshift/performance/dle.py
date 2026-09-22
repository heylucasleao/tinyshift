# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

"""Direct estimation of squared loss for regression or binary probabilities."""

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_array, check_is_fitted


class DirectLossEstimator(BaseEstimator):
    """Estimate model squared loss before current targets are available.

    A cloned regressor learns observed squared error from labeled reference
    data. Its inputs are the numeric features and the monitored model's
    prediction. The average estimated loss is MSE for regression. With binary
    labels encoded as 0 and 1 and predicted probabilities for class 1, it is
    the binary Brier score.

    Parameters
    ----------
    learner : sklearn-compatible regressor
        Unfitted model used to predict per-observation squared loss. It is
        cloned during :meth:`fit` and must provide ``fit`` and ``predict``.

    Attributes
    ----------
    loss_model_ : sklearn-compatible regressor
        Fitted clone of ``learner``.
    n_features_in_ : int
        Number of numeric input features supplied to :meth:`fit`, excluding
        the monitored model's prediction.

    Notes
    -----
    Estimated loss depends on the learned relationship between inputs and
    squared error remaining valid on current data. Current labels are not
    needed for :meth:`estimate`.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> dle = DirectLossEstimator(RandomForestRegressor(random_state=42))
    >>> dle.fit(X_reference, y_reference, predictions_reference)  # doctest: +ELLIPSIS
    DirectLossEstimator(...)
    >>> estimated_mse = dle.estimate(X_current, predictions_current)
    """

    def __init__(self, learner) -> None:
        self.learner = learner

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
        """Return observed squared loss for each reference observation.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Observed numeric targets.
        y_pred : array-like of shape (n_samples,)
            Monitored model predictions or binary class-1 probabilities.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Squared difference between each target and prediction.
        """
        predictions = self._vector(y_pred, "y_pred")
        target = self._target(y_true, len(predictions))
        return (target - predictions) ** 2

    def aggregate(self, losses) -> float:
        """Average nonnegative per-observation losses.

        Parameters
        ----------
        losses : array-like of shape (n_samples,)
            Finite, nonnegative squared losses.

        Returns
        -------
        float
            Mean loss, equivalent to MSE or binary Brier score.

        Raises
        ------
        ValueError
            If losses are negative, nonfinite, or not one-dimensional.
        """
        values = self._vector(losses, "losses")
        if np.any(values < 0):
            raise ValueError("losses must be nonnegative.")
        return float(np.mean(values))

    def fit(self, X, y_true, y_pred):
        """Fit a cloned learner on labeled reference observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric features used to predict loss.
        y_true : array-like of shape (n_samples,)
            Observed numeric targets.
        y_pred : array-like of shape (n_samples,)
            Monitored model predictions or binary class-1 probabilities.

        Returns
        -------
        DirectLossEstimator
            Fitted estimator.
        """
        features, predictions = self._inputs(X, y_pred)
        losses = self.observed_loss(y_true, predictions)
        self.loss_model_ = clone(self.learner)
        self.loss_model_.fit(np.column_stack((features, predictions)), losses)
        self.n_features_in_ = features.shape[1]
        return self

    def estimate_loss(self, X, y_pred) -> np.ndarray:
        """Predict nonnegative squared loss for each current observation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Current numeric features with the fitted feature count.
        y_pred : array-like of shape (n_samples,)
            Current predictions or binary class-1 probabilities.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Estimated squared loss, clipped at zero.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called.
        ValueError
            If the input shape is invalid or the learner returns nonfinite
            values or a number of values different from ``n_samples``.
        """
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
        """Estimate mean squared loss for an unlabeled current batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Current numeric features.
        y_pred : array-like of shape (n_samples,)
            Current predictions or binary class-1 probabilities.

        Returns
        -------
        float
            Estimated MSE, or binary Brier score for binary probabilities.
        """
        return self.aggregate(self.estimate_loss(X, y_pred))
