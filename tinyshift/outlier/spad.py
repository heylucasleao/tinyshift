# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
from sklearn.utils import check_array
from collections import Counter
from sklearn.decomposition import PCA
from .base import BaseHistogramModel
from typing import Union
from tinyshift.stats import StatisticalInterval


class SPAD(BaseHistogramModel):
    """
    SPAD (Statistical Probability Anomaly Detection) detects outliers by discretizing continuous data into bins and calculating anomaly scores based on the logarithm of inverse probabilities for each feature.

    SPAD+ enhances SPAD by incorporating Principal Components (PCs) from PCA, capturing feature correlations to detect multivariate anomalies (Type II Anomalies). The final score combines contributions from original features and PCs.

    Parameters
    ----------
    plus : bool, optional
        If True, applies PCA and concatenates transformed features. Default is False.

    Attributes
    ----------
    pca_model : PCA or None
        PCA model for dimensionality reduction if `plus` is True.
    plus : bool
        Indicates whether PCA is applied.

    References
    ----------
    Aryal, Sunil & Ting, Kai & Haffari, Gholamreza. (2016). Revisiting Attribute Independence Assumption in Probabilistic Unsupervised Anomaly Detection.
    https://www.researchgate.net/publication/301610958_Revisiting_Attribute_Independence_Assumption_in_Probabilistic_Unsupervised_Anomaly_Detection

    Aryal, Sunil & Agrahari Baniya, Arbind & Santosh, Kc. (2019). Improved histogram-based anomaly detector with the extended principal component features.
    https://www.researchgate.net/publication/336132587_Improved_histogram-based_anomaly_detector_with_the_extended_principal_component_features

    Notes
    -----
    - Lower SPAD scores indicate more anomalous observations (log-probabilities)
    - SPAD+ (plus=True) better detects multivariate anomalies by capturing feature correlations
    - Includes Laplace smoothing for probability estimation
    """

    def __init__(self, plus=False):
        """
        Parameters
        ----------
        plus : bool, optional
            If True, applies PCA and concatenates transformed features. Default is False.
        """
        self.pca_model = None
        self.plus = plus
        super().__init__()

    def fit(
        self,
        X: np.ndarray,
        nbins: Union[int, str] = "auto",
        random_state: int = 42,
        method="auto",
    ) -> "SPAD":
        """
        Fit the SPAD model to the input data.

        Parameters
        ----------
        X : np.ndarray
            Input data array of shape (n_samples, n_features).
        nbins : Union[int, str], optional
            Number of bins or binning strategy for discretizing continuous features. Options:
            - Integer: Exact number of bins for all continuous features.
            - String: Binning strategy, one of:
                - 'auto': Minimum of 'sturges' and 'fd' estimators.
                - 'fd': Freedman-Diaconis estimator (robust to outliers).
                - 'doane': Improved Sturges for non-normal data.
                - 'scott': Scott’s rule (efficient, less robust).
                - 'stone': Information-theoretic approach.
                - 'rice': Simple estimator based on sample size.
                - 'sturges': Optimal for Gaussian data.
                - 'sqrt': Square root of sample size.
            - Default is 'auto'. Set to an integer for fixed binning. Set 5 to replicate original paper.

        random_state : int, optional
            Random seed for reproducibility. Default is 42.
        method : str, optional
            Method to compute the interval for continuous features. Options:
            - "auto": Automatically selects the best method.
            - "stddev": Uses mean ± 3 standard deviations. (Original SPAD method)
            - "mad": Uses median ± 3*MAD.
            - "iqr": Uses median ± 1.5*IQR.
            - Callable: Custom function returning (lower, upper) bounds.
            - Tuple: Pre-defined (lower, upper) bounds.
            Default is "auto".

        Returns
        -------
        SPAD
            The fitted SPAD model instance.

        Notes
        -----
        - Extracts and stores feature data types and column names.
        - If `self.plus` is True, applies PCA and concatenates principal components (SPAD+).
        - For categorical features, computes relative frequencies with Laplace smoothing.
        - For continuous features, discretizes into bins and estimates probabilities.
        - Computes and stores anomaly scores in `self.decision_scores_`.
        """
        self._extract_feature_info(X)

        X = check_array(X)

        if self.plus:
            self.pca_model = PCA(random_state=random_state)
            self.pca_model = self.pca_model.fit(X)
            X = np.concatenate((X, self.pca_model.transform(X)), axis=1)
            self.feature_dtypes = np.concatenate(
                (self.feature_dtypes, np.array([np.float64] * len(self.feature_dtypes)))
            )

        _, self.n_features = X.shape

        for i in range(self.n_features):
            nbins = self._check_bins(X[:, i], nbins)

            if isinstance(self.feature_dtypes[i], pd.CategoricalDtype):
                value_counts = Counter(X[:, i])
                total_values = sum(value_counts.values())
                relative_frequencies = {
                    value: (count + 1) / (total_values + len(value_counts))
                    for value, count in value_counts.items()
                }
                self.feature_distributions.append(relative_frequencies)
            else:
                lower_bound, upper_bound = StatisticalInterval.compute_interval(
                    X[:, i], method
                )
                bin_edges = np.linspace(lower_bound, upper_bound, nbins + 1)
                digitized = np.digitize(X[:, i], bin_edges, right=True)
                unique_bins, counts = np.unique(digitized, return_counts=True)
                probabilities = (counts + 1) / (np.sum(counts) + len(unique_bins))
                self.feature_distributions.append([probabilities, bin_edges])

        self.decision_scores_ = self.decision_function(X)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values for the input data.
        """

        self._check_columns(X)

        X = check_array(X)
        outlier_scores = np.zeros(shape=(X.shape[0], self.n_features))

        if self.plus and X.shape[1] == self.n_features // 2:
            X = np.concatenate((X, self.pca_model.transform(X)), axis=1)

        for i in range(self.n_features):
            outlier_scores[:, i] = self._compute_outlier_score(X, i)

        return np.sum(outlier_scores, axis=1).ravel()
