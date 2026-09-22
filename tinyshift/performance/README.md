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
    fraction=0.5,
    chunk_size=100,  # close to the usual size of a current batch
    interval_method="stddev",
).fit(X_reference, y_reference, predictions_reference)

result = dle.predict(
    X_current, predictions_current, degradation_margin=0.10
)
print(result.relative_delta, result.reference_limit, result.degradation)
```

Reference rows retain their input order. The split occurs after
`floor(reference_size * (1 - fraction))` rows: the first part trains the loss
learner and the rest establish a held-out baseline. At least two rows are
required in each part. Each current batch (per ID in the analyzer) must also
contain at least two rows. For time series, order reference rows
chronologically before fitting. Use reference predictions made out of sample
by the monitored model when available, so the losses reflect its actual
prediction behavior.

`predict` returns a `DirectLossResult`:

| Field | Meaning |
|---|---|
| `reference_realized` | Observed mean squared loss on held-out reference rows |
| `reference_estimated` | Learner's estimated mean loss on those same rows |
| `reference_size` | Number of held-out reference rows |
| `current_estimated` | Learner's estimated mean loss on current rows |
| `estimated_delta` | `current_estimated - reference_estimated` |
| `relative_delta` | `current_estimated / reference_estimated - 1`; infinite for a positive current loss when reference loss is zero |
| `degradation_margin` | Minimum relevant relative increase; `0.10` means 10% |
| `reference_limit` | Upper bound of the interval over reference chunk mean losses |
| `degradation` | Whether `relative_delta > degradation_margin` and `current_estimated > reference_limit` |
| `current_size` | Number of current rows |

The two estimated values are compared so that the delta is measured on the
same learned scale. `reference_realized` shows how the learner performed on
held-out labeled data. `estimate(X_current, predictions_current)` returns only
the numeric current loss estimate; `estimate_loss(...)` returns one estimated
loss per row.

The held-out reference is divided into
`max(1, reference_size // chunk_size)` contiguous, nearly equal chunks. Here
`reference_size` is the number of **held-out** rows, not all rows supplied to
`fit`. When the holdout has at least `chunk_size` rows, each chunk has at
least that many rows; otherwise the entire holdout becomes one smaller chunk.
DLE computes the mean **estimated** loss of each chunk and passes those means
to `StatisticalInterval`. The default `"stddev"` method sets the upper limit
to the mean of chunk losses plus three standard deviations. `"mad"` and
`"iqr"` are also supported.

Choose `chunk_size` close to the expected current batch size. For example,
1,200 reference rows with `fraction=0.5` and `chunk_size=100` produce six
100-row held-out chunks, suitable for comparison with a current batch of
roughly 100 rows. The default is `chunk_size=50`. With one chunk, the limit
equals its mean and cannot describe historical variability; several chunks
are needed for that purpose. If current batch sizes vary substantially, the
fixed reference limit is less directly comparable across batches.

An alert requires both an increase above `degradation_margin` and current
estimated mean loss above the reference limit. The decision is deterministic
for the same fitted estimator and current batch; it is not a hypothesis test.
Relative changes can be numerically large when the reference mean loss is
close to zero, even if the absolute change is small. Inspect `estimated_delta`
alongside `relative_delta` in that case.

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
result = analyzer.predict(current_df, degradation_margin=0.10)
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
labeled reference (X, y, y_pred)
    ├── first rows: observed squared loss ──> fit loss learner
    └── held-out rows ──> estimated per-row losses
                            ├── mean of all rows ──> reference_estimated
                            └── means by chunk ──> StatisticalInterval
                                                    └── reference_limit

current (X, y_pred) ──> estimated per-row losses
                            └── mean ──> current_estimated

relative_delta = current_estimated / reference_estimated - 1
                 (if reference_estimated = 0: 0 for current = 0, else infinity)
degradation = relative_delta > degradation_margin
              and current_estimated > reference_limit
```

A positive `estimated_delta` means estimated loss increased relative to the
estimated reference baseline. `degradation` means the relative increase
exceeded both the chosen margin and the reference chunk limit. A `False`
result does not establish that performance is unchanged. The alert concerns
**estimated** loss and does not confirm that realized performance changed.
DLE needs the learned relationship between inputs and loss to remain useful on
current data. For classification, changed probability calibration can break
that relationship. Compare estimates with realized loss as current labels
arrive.

For runnable regression, binary probability, and panel examples, see
[`dle.ipynb`](../examples/dle.ipynb).
