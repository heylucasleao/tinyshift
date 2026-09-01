# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_array
from tinyshift.stats import StatisticalInterval


class RobustGaussianScaler(BaseEstimator, TransformerMixin):
    """A robust feature scaler that combines winsorization, power transformation and standardization.

    This transformer applies a three-step normalization process designed to handle outliers and
    non-Gaussian distributions:
    1. Winsorization (clipping extreme values based on quantiles)
    2. Power transformation (Yeo-Johnson or Box-Cox) to normalize data distribution
    3. Standard scaling (zero mean and unit variance)

    The combination of these steps makes the scaler robust to outliers while effectively
    normalizing the feature distributions.

    Attributes
    ----------
    power_transformer_ : PowerTransformer
        The fitted PowerTransformer instance.
    scaler_ : StandardScaler or None
        The fitted StandardScaler instance if standardize=True, None otherwise.
    lower_bounds_ : ndarray of shape (n_features,)
        The lower quantile values used for winsorization for each feature.
    upper_bounds_ : ndarray of shape (n_features,)
        The upper quantile values used for winsorization for each feature.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit. Only present when input is a pandas DataFrame.
    winsorization_bounds_ : list of tuples
        Property that returns the winsorization bounds for each feature as (lower, upper) tuples.

    """

    def __init__(
        self,
    ):
        self.power_transformer_ = None
        self.scaler_ = StandardScaler()
        self.lower_bounds_ = None
        self.upper_bounds_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(
        self,
        X: np.ndarray,
        winsorize_method: str = "stddev",
        power_method: str = "yeo-johnson",
    ) -> "RobustGaussianScaler":
        """Compute the winsorization bounds, power transform and scaling parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the transformation parameters.
        winsorize_method : str, default="stddev"
            The method used to compute winsorization bounds. Passed to StatisticalInterval.compute_interval().
            Common options include "stddev", "mad", "iqr", etc.
        power_method : {'yeo-johnson', 'box-cox'}, default='yeo-johnson'
            The power transformation method:
            - 'yeo-johnson': works for both positive and negative values
            - 'box-cox': only works for strictly positive data
        Returns
        -------
        self : object
            Fitted scaler.
        """
        self.feature_names_in_ = getattr(X, "columns", None)
        X = check_array(X, ensure_2d=False, dtype=np.float64, copy=True)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        self.n_features_in_ = X.shape[1]

        if power_method not in ["yeo-johnson", "box-cox"]:
            raise ValueError("power_method must be either 'yeo-johnson' or 'box-cox'")

        winsorize_bounds = np.array(
            [
                StatisticalInterval.compute_interval(X[:, i], method=winsorize_method)
                for i in range(self.n_features_in_)
            ]
        )

        self.power_transformer_ = PowerTransformer(
            method=power_method, standardize=False
        )

        self.lower_bounds_ = np.nan_to_num(winsorize_bounds[:, 0], nan=-np.inf)
        self.upper_bounds_ = np.nan_to_num(winsorize_bounds[:, 1], nan=np.inf)
        X = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        self.power_transformer_.fit(X)
        X = self.power_transformer_.transform(X)
        self.scaler_.fit(X)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the learned transformation to new data."""
        check_is_fitted(self)
        X = check_array(X, ensure_2d=False, dtype=np.float64, copy=True)
        X = X.reshape(-1, 1) if len(X.shape) == 1 else X

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        X = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        X = self.power_transformer_.transform(X)
        X = self.scaler_.transform(X)

        return X

    def fit_transform(
        self,
        X: np.ndarray,
        winsorize_method: str = "stddev",
        power_method: str = "yeo-johnson",
    ) -> np.ndarray:
        """Convenience method for fit().transform()."""

        return self.fit(X, winsorize_method, power_method).transform(X)

    @property
    def winsorization_bounds_(self) -> list[tuple[float, float]]:
        """Get the winsorization bounds for each feature as list of (lower, upper) tuples."""
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_"])
        return list(zip(self.lower_bounds_, self.upper_bounds_))
