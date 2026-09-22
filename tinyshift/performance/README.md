# Performance estimation

`tinyshift.performance` estimates a model's mean squared loss before targets
for the current batch are available. `DirectLossEstimator` (DLE) learns the
relationship between numeric features, model predictions, and observed squared
error in labeled reference data. It then estimates loss for new observations.

The same calculation covers two tasks:

| Task | `y` in the reference | `y_pred` in both batches | Mean loss |
|---|---|---|---|
| Regression | Numeric target | Numeric prediction | MSE |
| Binary classification | Label encoded as 0 or 1 | Probability of class 1 | Brier score |

For binary classification, supply valid probabilities in `[0, 1]` and 0/1
labels. DLE treats these as numeric values; it does not check that they are
probabilities or recalibrate them. Multiclass probability matrices are not
supported.

## Individual estimator

The estimator accepts a two-dimensional array of numeric features, a vector
of reference targets, and a vector of model predictions. Its `learner` is a
scikit-learn compatible regressor that predicts **per-observation squared
loss**. DLE clones the learner when fitting, so the supplied instance remains
unfitted.

```python
from sklearn.ensemble import RandomForestRegressor
from tinyshift.performance import DirectLossEstimator

dle = DirectLossEstimator(
    learner=RandomForestRegressor(random_state=42),
    fraction=0.25,
).fit(X_reference, y_reference, predictions_reference)

result = dle.predict(X_current, predictions_current)
print(result.reference_realized, result.current_estimated, result.degradation)
```

Reference rows retain their input order. The first `1 - fraction` train the
loss learner, while the last `fraction` establish a held-out baseline. At least
two training rows and one held-out row are required. For time series, order
reference rows chronologically before fitting. Use reference predictions made
out of sample by the monitored model when available, so the losses reflect its
actual prediction behavior.

`predict` returns a `DirectLossResult`:

| Field | Meaning |
|---|---|
| `reference_realized` | Observed mean squared loss on held-out reference rows |
| `reference_estimated` | Learner's estimated mean loss on those same rows |
| `reference_size` | Number of held-out reference rows |
| `current_estimated` | Learner's estimated mean loss on current rows |
| `estimated_delta` | `current_estimated - reference_estimated` |
| `degradation` | Whether `estimated_delta > 0` |
| `current_size` | Number of current rows |

The two estimated values are compared so that the delta is measured on the
same learned scale. `reference_realized` shows how the learner performed on
held-out labeled data. `estimate(X_current, predictions_current)` returns only
the numeric current loss estimate; `estimate_loss(...)` returns one estimated
loss per row.

To identify rows whose **predicted** loss is unusually high, DLE stores the
learner's per-row predictions on the held-out reference. `flag_high_loss`
compares current predicted losses with a quantile of those held-out predictions:

```python
flags = dle.flag_high_loss(X_current, predictions_current, quantile=0.99)
```

The result is one boolean per current row. It requires no current targets.
The threshold uses predicted losses on both sides of the comparison, and rows
equal to the threshold are not flagged. A flag is a forecast of high error,
not an observed outlier or a statistical test. With a small held-out sample,
high quantiles have limited resolution.

## Panel analyzer

`DirectLossAnalyzer` clones and fits one DLE for each `unique_id`, then runs
each current ID against its own reference and collects the `DirectLossResult`
objects in a DataFrame. It is an orchestration layer; the estimator performs
the split, baseline calculation, and comparison for each ID.

```python
from tinyshift.performance import DirectLossAnalyzer

analyzer = DirectLossAnalyzer(fraction=0.25).fit(
    reference_df,
    feature_cols=["feature_a", "feature_b"],
    id_col="unique_id",
    target_col="y",
    prediction_col="y_pred",
)
result = analyzer.predict(current_df)
```

The reference frame needs the ID, features, target, and prediction columns.
The current frame needs the same ID, features, and prediction columns, but no
target. `predict` accepts `id_col` and `prediction_col` overrides for a current
frame whose column names differ. Current IDs without a fitted reference raise
an error; reference IDs missing from a current batch are omitted. Rows retain
their input order within each ID, including for the reference split.

`predict` returns one row per current ID, in first-appearance order, with the
same fields as `DirectLossResult` plus the ID. `results_` holds one result
object per ID, and `summary()` returns the latest result table.

## Interpretation and monitoring flow

```text
labeled reference: X, y, y_pred
              │
              ├── first 1 - fraction ──> fit learner on (y - y_pred)²
              │
              └── last fraction ───────> observed and estimated baseline loss
                                              │
current: X, y_pred ──> estimated current loss ─┴─> estimated_delta
                                                     │
                                            estimated_delta > 0?
                                               │          │
                                              yes         no
                                          degradation   no increase
```

A positive `estimated_delta` means estimated loss increased relative to the
estimated reference baseline. `degradation` is **not** a p-value or a
statistical test, and it does not establish that realized performance changed.
DLE needs the learned relationship between inputs and loss to remain useful on
current data. For classification, changed probability calibration can break
that relationship. Compare estimates with realized loss as current labels
arrive.

For runnable regression, binary probability, and panel examples, see
[`dle.ipynb`](../examples/dle.ipynb).
