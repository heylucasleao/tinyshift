# Performance estimation

`DirectLossEstimator` learns squared error from labeled reference data and
estimates it for current observations without their targets. For regression,
the average squared error is MSE. For binary classification, provide labels 0/1
and the predicted probability of class 1; the same calculation is the binary
Brier score. It uses numeric features and the prediction to model per-row loss.

```python
from tinyshift.performance import DirectLossEstimator

dle = DirectLossEstimator().fit(X_reference, y_reference, predictions_reference)
estimated_mse = dle.estimate(X_current, predictions_current)
```

`DirectLossAnalyzer` fits one cloned estimator per panel ID. Each reference ID
needs at least two fitting rows and one held-out row. The last
`validation_fraction` of each ID forms the baseline, so order rows within an
ID as intended. Current targets are not required.

```python
from tinyshift.performance import DirectLossAnalyzer

analyzer = DirectLossAnalyzer().fit(
    reference_df, feature_cols=["feature_a", "feature_b"]
)
result = analyzer.predict(current_df)
```

Both frames need `unique_id`, the listed feature columns, and `y_pred`. The
reference also needs `y`. For binary classification, `y_pred` must be the
probability of class 1, and `y` must be 0 or 1. Column names can be changed in
`fit`. Results include held-out observed and estimated reference loss, current
estimated loss, their difference, and a `degradation` indicator for a positive
difference. This is an estimate, not a significance test; compare it against
realized loss when current labels arrive.
