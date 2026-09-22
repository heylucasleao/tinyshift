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
    alpha=0.05,
    n_resamples=999,
    random_state=42,
).fit(X_reference, y_reference, predictions_reference)

result = dle.predict(X_current, predictions_current)
print(result.current_estimated, result.p_value, result.degradation)
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
| `threshold` | Permutation critical value expressed as current mean estimated loss |
| `p_value` | One-sided Monte Carlo p-value for increased estimated loss |
| `degradation` | Whether `estimated_delta > 0` and `p_value <= alpha` |
| `current_size` | Number of current rows |

The two estimated values are compared so that the delta is measured on the
same learned scale. `reference_realized` shows how the learner performed on
held-out labeled data. `estimate(X_current, predictions_current)` returns only
the numeric current loss estimate; `estimate_loss(...)` returns one estimated
loss per row.

For each `predict` call, DLE pools the held-out and current **estimated**
per-row losses. It randomly permutes group assignments `n_resamples` times,
preserving both sample sizes, and recalculates the difference between current
and reference mean loss. The one-sided p-value uses the plus-one correction:

```python
(1 + number_of_permuted_deltas_at_least_observed) / (n_resamples + 1)
```

The `threshold` is `reference_estimated` plus the `1 - alpha` quantile of
permuted differences. The decision follows `p_value <= alpha`; with finite
permutations, it need not agree exactly with comparing against the displayed
threshold. Set `random_state` to reproduce the permutation result. As in
`ConDrift`, the test assumes observations are exchangeable under the null
hypothesis; temporal dependence can violate that assumption.

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
                                pool reference and current losses
                                              │
                                    permute group assignments
                                              │
current: X, y_pred ──> estimated current loss ─┴─> delta and p_value
                                              │
                                 delta > 0 and p_value <= alpha?
                                               │          │
                                              yes         no
                                          degradation   no increase
```

A positive `estimated_delta` means estimated loss increased relative to the
estimated reference baseline. `degradation` means that increase passed the
one-sided permutation test. The p-value concerns **estimated** loss; it does
not confirm that realized performance changed. Repeated monitoring can also
produce alerts by chance, even if the reference regime remains stable.
DLE needs the learned relationship between inputs and loss to remain useful on
current data. For classification, changed probability calibration can break
that relationship. Compare estimates with realized loss as current labels
arrive.

For runnable regression, binary probability, and panel examples, see
[`dle.ipynb`](../examples/dle.ipynb).
