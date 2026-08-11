# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import check_array
from sklearn.base import BaseEstimator
from typing import Union, List
import pandas as pd


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
    n_components : int
        Number of PCA components to keep (the component with the lowest explained variance will be discarded).

    Attributes
    ----------
    decision_scores_ : ndarray of shape (n_samples,)
        Reconstruction error scores after fitting.
    PCA : sklearn.decomposition.PCA
        Internal PCA instance configured with n_components-1.
    """

    def __init__(self) -> None:
        self.PCA = None
        self.decision_scores_: np.ndarray = None

    def _get_index(self, X: Union[pd.Series, List[np.ndarray], List[list]]):
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

    def fit(self, X: np.ndarray, n_components: int = None) -> "PCAReconstructionError":
        """
        Fit the model to the data and calculate reconstruction scores.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        n_components : int, optional
            Number of PCA components to keep (the component with the lowest explained variance will be discarded).
            If none, defaults to n_features - 1.

        Returns
        -------
        self : PCAReconstructionError
            The fitted detector.
        """
        X = check_array(X)
        if n_components is None:
            n_components = X.shape[1] - 1

        self.PCA = PCA(n_components=n_components)
        self.PCA.fit(X)
        X_reconstructed = self.PCA.inverse_transform(self.PCA.transform(X))
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
        if self.PCA is None:
            raise ValueError("Model must be fitted before prediction.")

        X = check_array(X)
        X_reconstructed = self.PCA.inverse_transform(self.PCA.transform(X))
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

        if self.PCA is None:
            raise ValueError("Model must be fitted before prediction.")
        index = self._get_index(X)
        X = check_array(X)
        scores = self.decision_function(X)
        threshold = np.quantile(self.decision_scores_, quantile, method="higher")
        return pd.Series(scores > threshold, index=index)
