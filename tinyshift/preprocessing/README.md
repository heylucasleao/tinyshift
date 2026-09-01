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

residualizer = FeatureResidualizer().fit(X_train)
X_residualized = residualizer.transform(X_train)

scaler = RobustGaussianScaler()
X_scaled = scaler.fit_transform(X_train)
```

Use these tools before fitting models when predictors are collinear, contain
strong outliers or have markedly non-Gaussian scales.
