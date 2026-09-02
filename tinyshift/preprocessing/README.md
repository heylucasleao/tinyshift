# Preprocessing (`tinyshift.preprocessing`)

Sklearn-compatible preprocessing utilities for robust machine-learning
workflows.

## Public API

- `filter_features_by_vif` iteratively removes features with high variance
  inflation factors.
- `FeatureResidualizer` replaces correlated predictors with regression
  residuals while retaining every feature.
- `RobustGaussianScaler` combines winsorization, a power transform and standard
  scaling.

```python
from tinyshift.preprocessing import (
    FeatureResidualizer,
    RobustGaussianScaler,
    filter_features_by_vif,
)

mask = filter_features_by_vif(X_train, threshold=5.0)
X_filtered = X_train[:, mask]

residualizer = FeatureResidualizer(corrcoef=0.8, corr_type="abs").fit(X_train)
X_residualized = residualizer.transform(X_train)

scaler = RobustGaussianScaler(
    winsorize_method="iqr",
    power_method="yeo-johnson",
)
X_scaled = scaler.fit_transform(X_train)
```

Transformer configuration belongs to the constructor, so both objects work
with `clone`, `Pipeline`, and parameter search. Calling `transform` before
`fit` raises `NotFittedError`. When fitted from a DataFrame, later DataFrames
must preserve the same columns and order.

`RobustGaussianScaler` returns a two-dimensional array, including for a
one-dimensional input. Its default IQR winsorization is robust to extremes;
Box-Cox can be selected explicitly but requires strictly positive values.

`filter_features_by_vif` requires a finite threshold of at least one. Constant
features receive infinite VIF and are candidates for removal. In underdetermined
settings, such as more predictors than observations, exact regressions can also
produce infinite VIFs, so feature removal may depend on input order when values
tie.

Use these tools before fitting models when predictors are collinear, contain
strong outliers or have markedly non-Gaussian scales.
