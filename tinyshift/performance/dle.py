# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

"""Direct estimation of squared loss for regression or binary probabilities."""

from dataclasses import dataclass
from numbers import Integral, Real

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
    relative_delta : float
        Estimated change relative to reference estimated loss. Infinite when
        reference loss is zero and current loss is positive.
    degradation_margin : float
        Minimum relative increase tested by the permutation test.
    p_value : float
        One-sided Monte Carlo p-value for an increase beyond the margin.
    degradation : bool
        Whether ``relative_delta`` exceeds the margin and ``p_value <= alpha``.
    current_size : int
        Number of current rows.
    """

    reference_estimated: float
    reference_realized: float
    reference_size: int
    current_estimated: float
    estimated_delta: float
    relative_delta: float
    degradation_margin: float
    p_value: float
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
    alpha : float, default=0.05
        Significance level for the one-sided permutation test of a relative
        increase beyond ``degradation_margin``. Must lie between zero and one.
    n_resamples : int, default=999
        Number of random permutations used to estimate the null distribution.
        Must allow a Monte Carlo p-value at or below ``alpha``.
    random_state : int or None, default=None
        Seed for reproducible permutations.

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
        Per-row losses predicted for the held-out reference. Used for
        permutation inference.

    Notes
    -----
    Estimated loss depends on the learned relationship between inputs and
    squared error remaining valid on current data. Current labels are not
    needed for :meth:`estimate`.

    Ordinary permutation inference assumes approximately independent
    observations. For serially dependent losses, inference may be
    anticonservative.

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

    def __init__(
        self,
        learner,
        fraction: float = 0.25,
        alpha: float = 0.05,
        n_resamples: int = 999,
        random_state: int | None = None,
    ) -> None:
        self.learner = learner
        self.fraction = fraction
        self.alpha = alpha
        self.n_resamples = n_resamples
        self.random_state = random_state

    def _validate_params(self) -> None:
        """Validate configuration used when fitting the reference."""
        if (
            isinstance(self.fraction, (bool, np.bool_))
            or not isinstance(self.fraction, Real)
            or not np.isfinite(self.fraction)
            or not 0 < self.fraction < 1
        ):
            raise ValueError("fraction must lie strictly between 0 and 1.")
        if (
            isinstance(self.alpha, (bool, np.bool_))
            or not isinstance(self.alpha, Real)
            or not np.isfinite(self.alpha)
            or not 0 < self.alpha < 1
        ):
            raise ValueError("alpha must lie strictly between 0 and 1.")
        if (
            isinstance(self.n_resamples, (bool, np.bool_))
            or not isinstance(self.n_resamples, Integral)
            or self.n_resamples < 1
        ):
            raise ValueError("n_resamples must be a positive integer.")
        if 1 / (self.n_resamples + 1) > self.alpha:
            raise ValueError("n_resamples is too small for the requested alpha.")

    @staticmethod
    def _validate_margin(degradation_margin: float) -> None:
        """Require a finite, nonnegative relative degradation margin."""
        if (
            isinstance(degradation_margin, (bool, np.bool_))
            or not isinstance(degradation_margin, Real)
            or not np.isfinite(degradation_margin)
            or degradation_margin < 0
        ):
            raise ValueError("degradation_margin must be finite and nonnegative.")

    def _reference_split(self, n_samples: int) -> int:
        """Return the fitting split after checking both reference partitions."""
        split = int(np.floor(n_samples * (1 - self.fraction)))
        if split < 2 or n_samples - split < 2:
            raise ValueError(
                "Reference needs at least two fitting rows and two held-out rows."
            )
        return split

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
        self._validate_params()
        features, predictions = self._inputs(X, y_pred)
        target = self._target(y_true, len(features))
        split = self._reference_split(len(features))
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

    @staticmethod
    def _welch_t_statistic(reference: np.ndarray, current: np.ndarray) -> float:
        """Return Welch's t statistic for current minus reference means."""
        difference = float(np.mean(current) - np.mean(reference))
        reference_var = float(np.var(reference, ddof=1)) if len(reference) > 1 else 0.0
        current_var = float(np.var(current, ddof=1)) if len(current) > 1 else 0.0
        standard_error = np.sqrt(
            reference_var / len(reference) + current_var / len(current)
        )
        if standard_error <= np.finfo(float).eps:
            return float(np.sign(difference) * np.inf) if difference else 0.0
        return difference / standard_error

    def _permutation_statistics(self, adjusted_current: np.ndarray) -> np.ndarray:
        """Generate Welch t statistics under permuted group assignments."""
        pooled = np.concatenate((self.reference_estimated_losses_, adjusted_current))
        rng = np.random.default_rng(self.random_state)
        statistics = np.empty(self.n_resamples, dtype=float)
        for index in range(self.n_resamples):
            permuted = pooled[rng.permutation(pooled.size)]
            statistics[index] = self._welch_t_statistic(
                permuted[: self.reference_size_], permuted[self.reference_size_ :]
            )
        return statistics

    def _calibrate(
        self,
        current_losses: np.ndarray,
        degradation_margin: float,
        relative_delta: float,
    ) -> tuple[float, bool]:
        """Test the relative increase using a permuted Welch t statistic."""
        adjusted_current = current_losses / (1.0 + degradation_margin)
        observed = self._welch_t_statistic(
            self.reference_estimated_losses_, adjusted_current
        )
        null_statistics = self._permutation_statistics(adjusted_current)
        p_value = float(
            (1 + np.count_nonzero(null_statistics >= observed)) / (self.n_resamples + 1)
        )
        degradation = relative_delta > degradation_margin and p_value <= self.alpha
        return p_value, bool(degradation)

    def _relative_delta(self, current_estimated: float) -> float:
        """Return relative loss change, including a zero-reference baseline."""
        if self.reference_estimated_ == 0:
            return float("inf") if current_estimated > 0 else 0.0
        return current_estimated / self.reference_estimated_ - 1.0

    def predict(self, X, y_pred, degradation_margin: float = 0.0) -> DirectLossResult:
        """Test whether current estimated loss exceeds a relative margin.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Current numeric features.
        y_pred : array-like of shape (n_samples,)
            Current predictions or binary class-1 probabilities.
        degradation_margin : float, default=0.0
            Nonnegative relative increase to test. For example, ``0.10``
            tests whether current mean estimated loss rose by more than 10%.

        Returns
        -------
        DirectLossResult
            Reference and current estimated losses, absolute and relative
            changes, tested margin, one-sided p-value, and alert indicator.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called.
        ValueError
            If ``degradation_margin`` is not finite and nonnegative, or if
            fewer than two current observations are supplied.

        Notes
        -----
        Dividing current losses by ``1 + degradation_margin`` transforms the
        boundary of the relative-margin hypothesis into equality of means.
        The permutation statistic is Welch's t statistic for the difference
        between group means. The plus-one correction gives the one-sided
        Monte Carlo p-value.
        Exact finite-sample validity requires exchangeability under the null;
        with unequal variances, studentization supports an asymptotic
        approximation for independent observations. This tests *predicted*
        loss and does not confirm realized degradation.
        """
        check_is_fitted(self, "reference_estimated_")
        self._validate_margin(degradation_margin)
        current_losses = self.estimate_loss(X, y_pred)
        current_size = len(current_losses)
        if current_size < 2:
            raise ValueError("Current needs at least two observations.")
        current_estimated = self.aggregate(current_losses)
        delta = current_estimated - self.reference_estimated_
        relative_delta = self._relative_delta(current_estimated)
        p_value, degradation = self._calibrate(
            current_losses, degradation_margin, relative_delta
        )
        return DirectLossResult(
            reference_estimated=self.reference_estimated_,
            reference_realized=self.reference_realized_,
            reference_size=self.reference_size_,
            current_estimated=current_estimated,
            estimated_delta=delta,
            relative_delta=relative_delta,
            degradation_margin=float(degradation_margin),
            p_value=p_value,
            degradation=degradation,
            current_size=current_size,
        )
