# Performance estimation

`DirectLossEstimator` estimates a regression model's loss when current targets
are not yet available. It learns from labeled reference observations, using
numeric features and the monitored model's prediction to predict a nonnegative
loss for each row. It supports MAE, MSE, and RMSE; RMSE is the square root of
the mean estimated squared loss.

```python
from tinyshift.performance import DirectLossEstimator

dle = DirectLossEstimator(metric="mae").fit(
    X_reference, y_reference, predictions_reference
)
estimated_mae = dle.estimate(X_current, predictions_current)
```

`DirectLossAnalyzer` fits one cloned estimator per ID. Each reference ID needs
at least two fitting rows and one held-out row. The last
`validation_fraction` of each ID forms the baseline, so rows should be ordered
within ID in the intended reference order. Current targets are not required.

```python
from tinyshift.performance import DirectLossAnalyzer, DirectLossEstimator

analyzer = DirectLossAnalyzer(DirectLossEstimator(metric="mae"))
analyzer.fit(reference_df, feature_cols=["feature_a", "feature_b"])
result = analyzer.predict(current_df)
```

Both frames need `unique_id`, the listed feature columns, and `y_pred`. The
reference also needs `y`. Column names can be changed in `fit`. Results include
held-out observed and estimated reference metrics, the current estimated
metric, their estimated difference, and a `degradation` indicator for a
positive difference. A positive difference is not a significance test and does
not verify realized degradation. Estimation depends on the loss model remaining
accurate on current data; compare estimates with observed metrics as labels
arrive. This module does not use the drift detectors.
