# Performance estimation

`DirectLossEstimator` learns squared error from labeled reference data and
estimates it for current observations without their targets. For regression,
the average squared error is MSE. For binary classification, provide labels 0/1
and the predicted probability of class 1; the same calculation is the binary
Brier score. It uses numeric features and the prediction to model per-row loss.
The first `1 - fraction` of reference rows fit the loss learner; the last
`fraction` establish a held-out baseline.

```python
from sklearn.ensemble import RandomForestRegressor
from tinyshift.performance import DirectLossEstimator

dle = DirectLossEstimator(learner=RandomForestRegressor(), fraction=0.25).fit(
    X_reference, y_reference, predictions_reference
)
result = dle.predict(X_current, predictions_current)
estimated_mse = result.current_estimated
```

`predict` returns a `DirectLossResult` with held-out observed and estimated
reference loss, current estimated loss, their difference, sample sizes, and a
`degradation` flag. `estimate` remains available when only the numeric loss
estimate is needed.

`DirectLossAnalyzer` clones and runs the estimator independently for each panel
ID, then collects the results in a DataFrame. Each reference ID needs at least
two fitting rows and one held-out row. Order rows within each ID as intended.
Current targets are not required.

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
`fit`. `degradation` means a positive change in estimated loss. This is an
estimate, not a significance test; compare it against realized loss when
current labels arrive.
