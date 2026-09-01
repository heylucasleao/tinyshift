# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_array


class FeatureResidualizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        """Feature transformer that reduces multicollinearity by residualizing highly correlated features.

        For each feature strongly correlated (above `corrcoef` threshold) with others, this class:
        1. Fits a linear regression model using the correlated features as predictors.
        2. Replaces the original feature with residuals (observed - predicted) from the model,
        effectively removing linear dependencies.

        Useful as a preprocessing step for linear models where multicollinearity is problematic.

        Attributes
        ----------
        models_ : Dict[int, Dict[str, Any]]
            Dictionary storing residualization models for each processed feature.
            Keys are feature indices; values are dicts with:
            - "model": Fitted `LinearRegression` object.
            - "features": Indices of features used as predictors.
        feature_names_in_ : Optional[np.ndarray]
            Names of input features if provided in a pandas DataFrame.
        n_features_in_ : Optional[int]
            Number of features seen during fit.

        Notes
        -----
        - Computes absolute Pearson correlations between all feature pairs (ignoring self-correlations)
        - Processes features in descending order of total correlation with others
        - For each target feature:
          - Selects correlated predictors (≥ threshold) not yet residualized
          - Fits linear model: target ~ predictors
          - Replaces target feature with model residuals
          - Excludes residualized features from future predictor sets
        - Preserves:
          - Feature order and non-correlated features
          - Non-linear relationships
        - Prevents circular dependencies by marking residualized features
        - Only removes linear relationships between features

        """
        self.models_ = None
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X: np.ndarray, corrcoef: float = 0.8, corr_type: str = "abs"):
        """
        Identify feature pairs with absolute correlation ≥ `corrcoef` and prepare residualization models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. Can be a pandas DataFrame (preserves column names) or numpy array.
        corrcoef : float, default=0.8
            Absolute correlation threshold for triggering residualization.
            Features with |ρ| ≥ this value will be residualized.
        corr_type : str, default="abs"
            Type of correlation to consider:
            - "abs": absolute correlation (default)
            - "pos": only positive correlation

        Returns
        -------
        self : FeatureResidualizer
            Fitted transformer.
        """

        if not 0 <= corrcoef <= 1:
            raise ValueError("corrcoef must be between 0 and 1")

        if corr_type not in ["abs", "pos"]:
            raise ValueError("corr_type must be either 'abs' or 'pos'")

        self.models_ = {}
        self.feature_names_in_ = getattr(X, "columns", None)
        X = check_array(X, ensure_2d=True, dtype=np.float64, copy=True)
        self.n_features_in_ = X.shape[1]
        corr = np.corrcoef(X, rowvar=False)

        if corr_type == "abs":
            corr = np.abs(corr)
            processing_order = np.argsort(-np.sum(corr, axis=1))
        elif corr_type == "pos":
            corr = np.where(corr < 0, 0, corr)
            processing_order = np.argsort(-np.sum(corr, axis=1))

        np.fill_diagonal(corr, 0)
        residualized = []

        for i in processing_order:
            corr_feature = corr[i]
            mask = np.ones(corr_feature.shape, dtype=bool)
            mask[residualized] = False
            indexes = np.argwhere((corr_feature >= corrcoef) & mask).flatten()

            if len(indexes) > 0:
                model = LinearRegression()
                model.fit(X[:, indexes], X[:, i])

                self.models_[i] = {
                    "model": model,
                    "features": indexes,
                }
                residualized.append(i)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply residualization to the input data using pre-trained models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform. Must match feature count of `fit()` input.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Data with residualized features (others remain unchanged).

        Raises
        ------
        ValueError
            If number of features in X doesn't match training data.
        """

        check_is_fitted(self, "models_")
        X = check_array(X, ensure_2d=True, dtype=np.float64, copy=True)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        for i, model_info in self.models_.items():
            model_info = self.models_[i]
            X[:, i] -= model_info["model"].predict(X[:, model_info["features"]])

        return X

    def fit_transform(
        self, X: np.ndarray, corrcoef: float = 0.8, corr_type: str = "abs"
    ) -> np.ndarray:
        """Convenience method for fit().transform()."""
        return self.fit(X, corrcoef=corrcoef, corr_type=corr_type).transform(X)
