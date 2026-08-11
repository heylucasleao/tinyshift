# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Union
import numpy as np
from sklearn.utils.validation import check_array
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression


def filter_features_by_vif(
    X: Union[np.ndarray, "pd.DataFrame"],
    threshold: float = 5,
    verbose: bool = False,
    n_jobs: int = -1,
) -> np.ndarray:
    """Filter features based on Variance Inflation Factor (VIF) to reduce multicollinearity.

    Iteratively removes features with VIF above the specified threshold until all remaining
    features have VIF below the threshold or only one feature remains. Uses parallel computation
    for efficient VIF calculation.

    Parameters
    ----------
    X : Union[np.ndarray, pd.DataFrame]
        Input feature matrix of shape (n_samples, n_features).
        Should be numeric and not contain missing values.
    threshold : float, optional
        VIF threshold for feature removal (default=15.0).
        Typical interpretation:
        - 1: No correlation
        - 1-5: Moderate correlation
        - 5-10: High correlation
        - >10: Very high correlation (consider removal)
    verbose : bool, optional
        If True, prints progress information during feature removal (default=False).
    n_jobs : int, optional
        Number of CPU cores to use for parallel computation (default=-1, all available cores).

    Returns
    -------
    np.ndarray
        Boolean mask of shape (n_features,) where:
        - True indicates the feature should be kept
        - False indicates the feature should be removed

    Raises
    ------
    ValueError
        If input validation fails (invalid threshold, empty array, insufficient samples/features).
    TypeError
        If input types are incorrect.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, _ = make_regression(n_samples=100, n_features=10, n_informative=5)
    >>> mask = filter_features_by_vif(X, threshold=5.0)
    >>> filtered_X = X[:, mask]
    """

    if not isinstance(threshold, (int, float)) or threshold < 1:
        raise ValueError("Threshold must be a numeric value >= 1.")
    if not isinstance(verbose, bool):
        raise TypeError("Verbose must be a boolean value.")
    if not isinstance(n_jobs, int):
        raise TypeError("n_jobs must be an integer.")

    feature_names_in_ = getattr(X, "columns", None)
    X = check_array(X, ensure_2d=True, dtype=np.float64, copy=True)

    if X.size == 0:
        raise ValueError("Input X cannot be empty.")
    if X.shape[0] < 2:
        raise ValueError("Input X must have at least two samples.")
    if X.shape[1] < 2:
        raise ValueError("Input X must have at least two features.")

    features = np.ones(X.shape[1], dtype=bool)

    def _vif(X: np.ndarray, i: int) -> float:
        """Helper function to compute VIF for a single feature."""
        y = X[:, i]
        X = np.delete(X, i, axis=1)
        model = LinearRegression().fit(X, y)
        r_squared = model.score(X, y)
        return 1.0 / (1.0 - r_squared) if r_squared < 1 else np.inf

    for _ in range(features.shape[0]):
        vif = np.zeros(features.shape[0])
        mask = np.where(features)[0]

        if len(mask) < 2:
            break

        vif[mask] = Parallel(n_jobs=n_jobs)(
            delayed(_vif)(X[:, features], i) for i in range(X[:, features].shape[1])
        )
        max_vif = np.max(vif)

        if max_vif <= threshold:
            break

        idx_max_vif = np.argmax(vif)
        features[idx_max_vif] = False

        if verbose:
            feature_name = (
                feature_names_in_[idx_max_vif]
                if feature_names_in_ is not None
                else idx_max_vif
            )
            print(f"Removing feature {feature_name} with VIF: {max_vif:.2f}")

    return features
