# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.utils import check_array


class PCAReconstructionError(BaseEstimator):
    """
    Noise PCA-based outlier detector.

    Uses PCA for outlier detection through data reconstruction by:
        1. Discarding the PCA component with the lowest covariance
        2. Reversing the PCA process to reconstruct the data
        3. Calculating reconstruction errors
        4. Identifying outliers as points with highest reconstruction error

    Parameters
    ----------
    n_components : int or None, default=None
        Number of PCA components to retain. ``None`` retains one fewer than the
        number of input features so that reconstruction error remains defined.

    Attributes
    ----------
    decision_scores_ : ndarray of shape (n_samples,)
        Reconstruction error scores after fitting.
    pca_ : sklearn.decomposition.PCA
        Internal fitted PCA instance.
    """

    def __init__(self, n_components: int | None = None) -> None:
        self.n_components = n_components

    def _get_index(self, X: pd.Series | list[np.ndarray] | list[list]):
        """
        Helper function to retrieve the index of a pandas Series or generate a default index.
        """
        return X.index if hasattr(X, "index") else list(range(len(X)))

    def _calculate_reconstruction_error(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> np.ndarray:
        """
        Calculate squared reconstruction error for each sample.

        Parameters
        ----------
        original : ndarray of shape (n_samples, n_features)
            Original data before transformation.
        reconstructed : ndarray of shape (n_samples, n_features)
            Data after reconstruction.

        Returns
        -------
        errors : ndarray of shape (n_samples,)
            Array of squared errors for each sample.
        """
        return np.sum((original - reconstructed) ** 2, axis=1)

    def fit(self, X: np.ndarray, y=None) -> "PCAReconstructionError":
        """
        Fit the model to the data and calculate reconstruction scores.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        Returns
        -------
        self : PCAReconstructionError
            The fitted detector.
        """
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        n_components = self.n_components
        if n_components is None:
            if self.n_features_in_ < 2:
                raise ValueError(
                    "PCAReconstructionError requires at least two features when "
                    "n_components is None."
                )
            n_components = self.n_features_in_ - 1
        if (
            isinstance(n_components, (bool, np.bool_))
            or not isinstance(n_components, (int, np.integer))
            or not 1 <= int(n_components) < self.n_features_in_
        ):
            raise ValueError("n_components must be an integer in [1, n_features - 1].")

        self.pca_ = PCA(n_components=int(n_components))
        self.pca_.fit(X)
        X_reconstructed = self.pca_.inverse_transform(self.pca_.transform(X))
        self.decision_scores_ = self._calculate_reconstruction_error(X, X_reconstructed)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error scores for each sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to evaluate.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Reconstruction error scores for each sample.
        """
        if not hasattr(self, "pca_"):
            raise ValueError("Model must be fitted before prediction.")

        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but PCAReconstructionError was "
                f"fitted with {self.n_features_in_} features."
            )
        X_reconstructed = self.pca_.inverse_transform(self.pca_.transform(X))
        return self._calculate_reconstruction_error(X, X_reconstructed)

    def predict(self, X: np.ndarray, quantile: float = 0.99) -> np.ndarray:
        """
        Identify outliers based on reconstruction error.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to evaluate.
        quantile : float, default=0.99
            Threshold quantile for outlier detection.

        Returns
        -------
        outliers : ndarray of shape (n_samples,)
            Boolean array indicating outliers (True) and inliers (False).

        Raises
        ------
        ValueError
            If model hasn't been fitted yet.

        Notes
        -----
        - The threshold is computed as the specified quantile of the reconstruction errors.
        - Higher reconstruction errors indicate more anomalous observations.
        """

        if not hasattr(self, "pca_"):
            raise ValueError("Model must be fitted before prediction.")
        if not np.isscalar(quantile) or not np.isfinite(quantile):
            raise ValueError("quantile must be finite and lie in [0, 1].")
        if not 0.0 <= float(quantile) <= 1.0:
            raise ValueError("quantile must be finite and lie in [0, 1].")
        index = self._get_index(X)
        scores = self.decision_function(X)
        threshold = np.quantile(self.decision_scores_, quantile, method="higher")
        return pd.Series(scores > threshold, index=index)
