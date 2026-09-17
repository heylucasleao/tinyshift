# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

"""Shared primitives for reference-to-current drift detection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


@dataclass(frozen=True)
class DriftResult:
    """Result of comparing a current sample with a fitted reference.

    Attributes
    ----------
    score : float
        Observed distribution distance. Retained for internal reporting;
        drift decisions are based on ``p_value``.
    threshold : float
        Empirical ``1 - alpha`` quantile of the permutation scores.
    p_value : float
        Monte Carlo permutation p-value with the plus-one correction.
    drift : bool
        Whether ``p_value`` is less than or equal to ``alpha``.
    reference_size : int
        Number of observations in the fitted reference.
    current_size : int
        Number of observations in the current sample.
    """

    score: float
    threshold: float | None
    p_value: float | None
    drift: bool | None
    reference_size: int
    current_size: int


class BaseDrift(BaseEstimator, ABC):
    """Shared lifecycle for vector-based two-sample drift detectors.

    Subclasses define sample validation and a distribution distance. The base
    class stores a fixed reference sample and tests each current sample through
    a Monte Carlo permutation test of distributional equality.

    Parameters
    ----------
    alpha : float
        Significance level used to classify drift. Must be strictly between
        zero and one.
    n_resamples : int
        Number of random label permutations used to approximate the null
        distribution. Must be positive.
    random_state : int or None
        Seed passed to NumPy's random generator. Use an integer for
        reproducible permutation results.
    min_reference_size : int
        Minimum number of observations accepted by :meth:`fit`.
    min_current_size : int
        Minimum number of observations accepted by :meth:`predict`.

    Attributes
    ----------
    reference_ : numpy.ndarray
        Validated one-dimensional reference sample. Created by :meth:`fit`.
    reference_size_ : int
        Number of fitted reference observations. Created by :meth:`fit`.

    Notes
    -----
    Under the null hypothesis, reference and current observations must be
    exchangeable. In the usual two-sample setting this requires independent
    observations drawn from the same distribution. Subclasses may recompute
    reference-dependent normalization inside every permutation.
    """

    def __init__(
        self,
        alpha: float,
        n_resamples: int,
        random_state: int | None,
        min_reference_size: int,
        min_current_size: int,
    ) -> None:
        self.alpha = alpha
        self.n_resamples = n_resamples
        self.random_state = random_state
        self.min_reference_size = min_reference_size
        self.min_current_size = min_current_size

    def _validate_params(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")
        if not isinstance(self.n_resamples, int) or self.n_resamples < 1:
            raise ValueError("n_resamples must be a positive integer.")
        for name, value in (
            ("min_reference_size", self.min_reference_size),
            ("min_current_size", self.min_current_size),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

    @abstractmethod
    def _validate_sample(self, values: Any, name: str) -> np.ndarray:
        """Return a validated one-dimensional sample."""

    @abstractmethod
    def _distance(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Compute the configured distance between two validated samples."""

    def fit(self, reference: Any) -> "BaseDrift":
        """Fit the detector on a reference sample.

        Parameters
        ----------
        reference : array-like of shape (n_observations,)
            Baseline observations. Accepted values and dtypes are defined by
            the concrete detector.

        Returns
        -------
        BaseDrift
            Fitted detector.

        Raises
        ------
        ValueError
            If detector parameters are invalid, the sample is not a valid
            one-dimensional input, or it contains fewer than
            ``min_reference_size`` observations.
        """
        self._validate_params()
        values = self._validate_sample(reference, "reference")
        if values.size < self.min_reference_size:
            raise ValueError(
                f"reference must contain at least {self.min_reference_size} observations."
            )
        self.reference_ = values.copy()
        self.reference_size_ = int(values.size)
        return self

    def _inference_distance(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Return the statistic for one permuted group assignment."""
        return self._distance(reference, current)

    def _permutation_scores(self, current: np.ndarray) -> np.ndarray:
        """Generate the null distribution through random label permutations.

        Parameters
        ----------
        current : numpy.ndarray of shape (n_current,)
            Validated current observations. They are pooled with
            :attr:`reference_` before group labels are permuted.

        Returns
        -------
        numpy.ndarray of shape (n_resamples,)
            Distance statistic obtained from every random group assignment.

        Notes
        -----
        Each permutation preserves the original reference and current sample
        sizes. The first ``reference_size_`` shuffled observations form the
        permuted reference and the remainder form the permuted comparison
        group. :meth:`_inference_distance` is evaluated after every split so
        subclasses can recompute reference-dependent quantities.

        The observed, unpermuted statistic is not inserted into the returned
        array. :meth:`predict` accounts for it with the plus-one Monte Carlo
        correction when computing the p-value.
        """
        pooled = np.concatenate((self.reference_, current))
        rng = np.random.default_rng(self.random_state)
        scores = np.empty(self.n_resamples, dtype=float)
        for index in range(self.n_resamples):
            permuted = pooled[rng.permutation(pooled.size)]
            reference = permuted[: self.reference_size_]
            comparison = permuted[self.reference_size_ :]
            scores[index] = self._inference_distance(reference, comparison)
        return scores

    def _calibrate(self, X: np.ndarray, score: float) -> tuple[float, float, bool]:
        """Derive the critical value, p-value, and decision from permutations."""
        null_scores = self._permutation_scores(X)
        threshold = float(np.quantile(null_scores, 1 - self.alpha, method="higher"))
        p_value = float(
            (1 + np.count_nonzero(null_scores >= score)) / (self.n_resamples + 1)
        )
        return threshold, p_value, p_value <= self.alpha

    def predict(self, X: Any) -> DriftResult:
        """Test a current sample for distribution drift.

        Parameters
        ----------
        X : array-like of shape (n_observations,)
            Current observations compared with the fitted reference. Accepted
            values and dtypes are defined by the concrete detector.

        Returns
        -------
        DriftResult
            Permutation-test result containing the observed distance, empirical
            critical value, Monte Carlo p-value, drift decision, and sample
            sizes.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called.
        ValueError
            If ``X`` is not a valid one-dimensional sample or contains fewer
            than ``min_current_size`` observations.

        Notes
        -----
        The null hypothesis is that the fitted reference and ``X`` come from
        the same distribution. The Monte Carlo p-value is

        ``(1 + number of permutation scores >= observed score) /``
        ``(n_resamples + 1)``.

        Drift is reported when this p-value is less than or equal to ``alpha``.
        The plus-one correction prevents zero p-values and yields a valid
        random-permutation test under exchangeability. Consequently, rejection
        at level ``alpha`` is possible only when
        ``n_resamples >= 1 / alpha - 1``.
        """
        check_is_fitted(self, "reference_")
        current = self._validate_sample(X, "current")
        if current.size < self.min_current_size:
            raise ValueError(
                f"current must contain at least {self.min_current_size} observations."
            )
        score = self._distance(self.reference_, current)
        threshold, p_value, drift = self._calibrate(current, score)
        return DriftResult(
            score=score,
            threshold=threshold,
            p_value=p_value,
            drift=drift,
            reference_size=self.reference_size_,
            current_size=int(current.size),
        )
