# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class BaseHistogramModel(BaseEstimator):
    """
    Base class for histogram-based models.

    Attributes:
    n_features : int or None
        Number of features in the dataset.

    feature_dtypes : list or None
        Data types of the features.

    feature_names : list
        List of column names.

    feature_distributions : list
        List of distributions for each feature. Each distribution can be a dictionary or a list of numpy arrays.

    decision_scores_ : array-like or None
        Decision scores for the samples.
    """

    def __init__(self):
        self.n_features = None
        self.feature_names = None
        self.feature_dtypes = []
        self.feature_distributions: list[dict | list[np.ndarray]] = []
        self.decision_scores_ = None

    def _check_bins(self, X: np.ndarray, nbins: int | str) -> int:
        """
        Determine the number of bins for histogram binning.
        Parameters:
        -----------
        X : np.ndarray
            The input data array for which the bins are to be determined.
        nbins : Union[int, str]
            The number of bins or a binning strategy. If an integer, it must be positive.
            If a string, it should be a valid binning strategy recognized by `np.histogram_bin_edges`.
        Returns:
        --------
        int
            The number of bins to be used for histogram binning.
        Raises:
        -------
        ValueError
            If `nbins` is not a positive integer or a valid binning strategy.
        """

        if (
            isinstance(nbins, (int, np.integer))
            and not isinstance(nbins, (bool, np.bool_))
            and nbins > 0
        ):
            return int(nbins)
        elif isinstance(nbins, str):
            try:
                bin_edges = np.histogram_bin_edges(X, bins=nbins)
                return len(bin_edges) - 1
            except ValueError as e:
                raise ValueError(
                    f"Invalid binning strategy '{nbins}'. Please use a positive integer or one of the following valid strategies: "
                    "'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.\n"
                    "Descriptions:\n"
                    "- 'auto': Minimum bin width between the 'sturges' and 'fd' estimators. Provides good all-around performance.\n"
                    "- 'fd' (Freedman Diaconis Estimator): Robust estimator that accounts for data variability and size.\n"
                    "- 'doane': Improved version of Sturges’ estimator for non-normal datasets.\n"
                    "- 'scott': Less robust estimator that considers data variability and size.\n"
                    "- 'stone': Based on leave-one-out cross-validation of the integrated squared error. Generalizes Scott’s rule.\n"
                    "- 'rice': Considers only data size, often overestimates the number of bins.\n"
                    "- 'sturges': Optimal for Gaussian data, underestimates bins for large non-Gaussian datasets.\n"
                    "- 'sqrt': Square root of data size, used for simplicity and speed."
                ) from e
        else:
            raise ValueError(
                "nbins must be a positive integer or a valid `np.histogram_bin_edges` binning strategy."
            )

    def _check_columns(self, X: np.ndarray | pd.DataFrame):
        """
        Check if the columns of the input data match the columns of the training data.

        Parameters:
        -----------
        X : Union[np.ndarray, pd.DataFrame]
            The input data array or DataFrame to be checked.

        Raises:
        -------
        ValueError
            If the columns of the input data do not match the columns of the training data.
        """
        if isinstance(X, pd.DataFrame) and X.columns.tolist() != self.feature_names:
            raise ValueError(
                "The columns of the input data do not match the columns of the training data."
            )

    def _reset_fit_state(self) -> None:
        """Clear learned distributions before fitting again."""
        self.feature_distributions = []
        self.decision_scores_ = None

    def _validate_n_features(self, X: np.ndarray) -> None:
        """Require the same feature count used during fitting."""
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self.__class__.__name__} "
                f"was fitted with {self.n_features_in_} features."
            )

    def _extract_feature_info(self, X: pd.Series | pd.DataFrame | np.ndarray):
        """
        Extract feature information from the input data.

        Parameters:
        -----------
        X : Union[pd.Series, pd.DataFrame]
            The input data from which to extract feature information.

        Raises:
        -------
        TypeError
            If the input data is not a pandas Series or DataFrame.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            self.feature_dtypes = X.dtypes.values
        elif isinstance(X, pd.Series):
            self.feature_names = [X.name] if X.name else ["feature_0"]
            self.feature_dtypes = [X.dtype]
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                self.feature_names = ["feature_0"]
                self.feature_dtypes = [X.dtype]
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                self.feature_dtypes = np.repeat(X.dtype, X.shape[1])
        else:
            raise TypeError(
                "Input data must be a pandas Series, DataFrame, or numpy ndarray."
            )

    def _get_index(self, X: pd.Series | list[np.ndarray] | list[list]):
        """
        Helper function to retrieve the index of a pandas Series or generate a default index.
        """
        return X.index if hasattr(X, "index") else list(range(len(X)))

    def _compute_outlier_score(self, X: np.ndarray, i: int) -> np.ndarray:
        """
        Calculates the self-information (surprisal) outlier score for each value in a specified feature column.

        This method quantifies how "surprising" or "rare" each value in the feature column is, based on its estimated probability.
        The self-information is computed as the negative natural logarithm of the probability of each value: -log(p).
        Higher scores indicate rarer (more outlier-like) values, while lower scores indicate more common values.

        Parameters:
            X (np.ndarray): The input data array of shape (n_samples, n_features).
            i (int): The index of the feature column for which to compute outlier scores.

        Returns:
            np.ndarray: An array of self-information scores (digits) for each value in the specified feature column.

        Notes:
            - For categorical features, probabilities are retrieved from a precomputed distribution dictionary.
              If a value is not found, a small probability (1e-9) is used to avoid log(0).
            - For continuous features, probabilities are estimated using histogram binning.
              Each value is assigned to a bin, and the corresponding bin probability is used.
            - A small constant (1e-9) is added to probabilities to ensure numerical stability and avoid taking the logarithm of zero.
        """

        if isinstance(self.feature_dtypes[i], pd.CategoricalDtype):
            densities = np.array(
                [self.feature_distributions[i].get(value, 1e-9) for value in X[:, i]]
            )
        else:
            probabilities, bin_edges = self.feature_distributions[i]
            values = X[:, i].astype(float)
            bin_indices = np.searchsorted(bin_edges, values, side="right") - 1
            bin_indices = np.clip(bin_indices, 0, len(probabilities) - 1)
            densities = probabilities[bin_indices]
            outside = (values < bin_edges[0]) | (values > bin_edges[-1])
            densities = np.where(outside, 1e-9, densities)

        return -np.log(densities + 1e-9)

    def predict(self, X: np.ndarray, quantile: float = 0.99) -> np.ndarray:
        """
        Identify outliers based on anomaly scores.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to evaluate.
        quantile : float, default=0.99
            Threshold quantile for outlier detection.

        Raises
        ------
        ValueError
            If model hasn't been fitted yet.

        Notes
        -----
        - Following the original HBOS paper, higher scores indicate more anomalous observations.
        """

        if self.decision_scores_ is None:
            raise ValueError("Model must be fitted before prediction.")
        if not np.isscalar(quantile) or not np.isfinite(quantile):
            raise ValueError("quantile must be finite and lie in [0, 1].")
        if not 0.0 <= float(quantile) <= 1.0:
            raise ValueError("quantile must be finite and lie in [0, 1].")
        index = self._get_index(X)
        scores = self.decision_function(X)
        threshold = np.quantile(self.decision_scores_, quantile, method="higher")
        return pd.Series(scores > threshold, index=index)
