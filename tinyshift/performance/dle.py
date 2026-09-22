# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

"""Direct estimation of squared loss for regression or binary probabilities."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_array, check_is_fitted


@dataclass(frozen=True)
class DirectLossResult:
    """Estimated change in mean squared loss from a held-out reference.

    Attributes
    ----------
    reference_estimated : float
        Estimated mean loss on held-out reference rows.
    reference_realized : float
        Observed mean loss on held-out reference rows.
    reference_size : int
        Number of held-out reference rows.
    current_estimated : float
        Estimated mean loss on current rows.
    estimated_delta : float
        Current estimated loss minus reference estimated loss.
    degradation : bool
        Whether ``estimated_delta`` is positive.
    current_size : int
        Number of current rows.
    """

    reference_estimated: float
    reference_realized: float
    reference_size: int
    current_estimated: float
    estimated_delta: float
    degradation: bool
    current_size: int


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
    fraction : float, default=0.25
        Fraction of the reference rows held out to establish the baseline.
        Must be strictly between zero and one.

    Attributes
    ----------
    loss_model_ : sklearn-compatible regressor
        Fitted clone of ``learner``.
    n_features_in_ : int
        Number of numeric input features supplied to :meth:`fit`, excluding
        the monitored model's prediction.
    reference_estimated_, reference_realized_ : float
        Estimated and observed mean squared loss on held-out reference rows.
    reference_size_ : int
        Number of held-out reference rows.
    reference_estimated_losses_ : numpy.ndarray
        Per-row losses predicted for the held-out reference. Used to set a
        threshold for unusually high predicted loss.

    Notes
    -----
    Estimated loss depends on the learned relationship between inputs and
    squared error remaining valid on current data. Current labels are not
    needed for :meth:`estimate`.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> dle = DirectLossEstimator(RandomForestRegressor(random_state=42))
    >>> result = dle.fit(X_reference, y_reference, predictions_reference).predict(
    ...     X_current, predictions_current
    ... )
    >>> result.current_estimated  # doctest: +SKIP
    ...
    """

    def __init__(self, learner, fraction: float = 0.25) -> None:
        self.learner = learner
        self.fraction = fraction

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
        """Fit on the first reference rows and establish a held-out baseline.

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

        Notes
        -----
        Input order determines the split. The first ``1 - fraction`` of rows
        fit the learner and the remaining rows establish the reference
        baseline. For time series, order rows before calling this method.
        """
        if (
            isinstance(self.fraction, (bool, np.bool_))
            or not isinstance(self.fraction, Real)
            or not np.isfinite(self.fraction)
            or not 0 < self.fraction < 1
        ):
            raise ValueError("fraction must lie strictly between 0 and 1.")
        features, predictions = self._inputs(X, y_pred)
        target = self._target(y_true, len(features))
        split = int(np.floor(len(features) * (1 - self.fraction)))
        if split < 2 or split == len(features):
            raise ValueError(
                "Reference needs at least two fitting rows and one held-out row."
            )
        losses = self.observed_loss(target[:split], predictions[:split])
        model = clone(self.learner)
        model.fit(np.column_stack((features[:split], predictions[:split])), losses)
        self.loss_model_ = model
        self.n_features_in_ = features.shape[1]
        holdout_losses = self.observed_loss(target[split:], predictions[split:])
        self.reference_realized_ = self.aggregate(holdout_losses)
        self.reference_estimated_losses_ = self.estimate_loss(
            features[split:], predictions[split:]
        )
        self.reference_estimated_ = self.aggregate(self.reference_estimated_losses_)
        self.reference_size_ = len(holdout_losses)
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

    def high_estimated_loss_mask(self, X, y_pred, quantile: float = 0.99) -> np.ndarray:
        """Flag rows with predicted loss above a held-out reference quantile.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Current numeric features.
        y_pred : array-like of shape (n_samples,)
            Current predictions or binary class-1 probabilities.
        quantile : float, default=0.99
            Quantile of held-out *predicted* losses used as the threshold.
            Must lie in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray of bool, shape (n_samples,)
            Whether each predicted loss is strictly above the threshold.

        Notes
        -----
        These flags identify unusually high **predicted** loss, not observed
        errors. Their reliability depends on the loss learner remaining useful
        for current data. Ties at the threshold are not flagged.
        """
        check_is_fitted(self, "reference_estimated_losses_")
        if (
            isinstance(quantile, (bool, np.bool_))
            or not isinstance(quantile, Real)
            or not np.isfinite(quantile)
            or not 0 <= quantile <= 1
        ):
            raise ValueError("quantile must be finite and lie in [0, 1].")
        threshold = np.quantile(
            self.reference_estimated_losses_, quantile, method="higher"
        )
        return self.estimate_loss(X, y_pred) > threshold

    def predict(self, X, y_pred) -> DirectLossResult:
        """Compare current estimated loss with the held-out reference.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Current numeric features.
        y_pred : array-like of shape (n_samples,)
            Current predictions or binary class-1 probabilities.

        Returns
        -------
        DirectLossResult
            Reference losses, current estimated loss, their difference, and
            the indicator for a positive difference.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called.

        Notes
        -----
        ``degradation`` is an estimated increase in loss, not a statistical
        test or confirmation of realized degradation.
        """
        check_is_fitted(self, "reference_estimated_")
        current_size = len(self._vector(y_pred, "y_pred"))
        current_estimated = self.estimate(X, y_pred)
        delta = current_estimated - self.reference_estimated_
        return DirectLossResult(
            reference_estimated=self.reference_estimated_,
            reference_realized=self.reference_realized_,
            reference_size=self.reference_size_,
            current_estimated=current_estimated,
            estimated_delta=delta,
            degradation=bool(delta > 0),
            current_size=current_size,
        )
