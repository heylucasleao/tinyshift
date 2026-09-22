# Performance estimation

## Regression: direct loss estimation

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

## Classification: probability-based estimation

`ConfidenceBasedPerformanceEstimator` estimates a confusion matrix from
calibrated class probabilities. Each observation is assigned to its most
probable predicted class; its probabilities contribute expected counts across
the possible true classes. Accuracy is the expected diagonal share. Precision,
recall, and F1 are calculated from the expected confusion matrix. Binary
metrics use the last class as positive by default; multiclass metrics use macro
averaging by default.

```python
from tinyshift.performance import (
    ConfidenceBasedPerformanceAnalyzer,
    ConfidenceBasedPerformanceEstimator,
)

analyzer = ConfidenceBasedPerformanceAnalyzer(
    ConfidenceBasedPerformanceEstimator(metric="accuracy")
)
analyzer.fit(
    reference_df,
    probability_cols={"low": "p_low", "medium": "p_medium", "high": "p_high"},
)
result = analyzer.predict(current_df)
```

Supply one probability column per class, including both classes for binary
classification. Each row must contain values in `[0, 1]` summing to one.
`reference_df` needs `unique_id`, `y`, and the probability columns;
`current_df` needs only `unique_id` and the probability columns. The reference
predictions should be out-of-sample predictions from the monitored classifier.
The analyzer fits an independent estimator for every ID and reports observed
and estimated reference metrics, estimated current performance, and their
difference. For these metrics, `degradation` means a *decrease* in estimated
performance. It is not a statistical significance test.

The current probabilities must remain calibrated. This implementation assumes
they are already calibrated; it does not fit a calibrator. A shift in the
relationship between probabilities and true labels can make estimated
performance inaccurate even when the output looks stable. Verify estimates
against realized metrics when labels arrive.
