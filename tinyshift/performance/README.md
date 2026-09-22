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

result = dle.predict(
    X_current, predictions_current, degradation_margin=0.10
)
print(result.relative_delta, result.p_value, result.degradation)
```

Reference rows retain their input order. The first `1 - fraction` train the
loss learner, while the last `fraction` establish a held-out baseline. At least
two training rows and two held-out rows are required. Each current batch (per
ID in the analyzer) must also contain at least two rows. For time series, order
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
| `relative_delta` | `current_estimated / reference_estimated - 1`; infinite for a positive current loss when reference loss is zero |
| `degradation_margin` | Minimum relative increase tested; `0.10` means 10% |
| `p_value` | One-sided Monte Carlo p-value for exceeding the margin |
| `degradation` | Whether `relative_delta > degradation_margin` and `p_value <= alpha` |
| `current_size` | Number of current rows |

The two estimated values are compared so that the delta is measured on the
same learned scale. `reference_realized` shows how the learner performed on
held-out labeled data. `estimate(X_current, predictions_current)` returns only
the numeric current loss estimate; `estimate_loss(...)` returns one estimated
loss per row.

`degradation_margin` changes the hypothesis being tested. With a margin `m`,
the null is that current mean estimated loss has increased by **at most** `m`
relative to reference. DLE divides current per-row losses by `1 + m`, then
compares their adjusted mean with the reference mean. Welch's t statistic
divides the difference between means by its estimated standard error.
Relative changes can be numerically large when the reference mean loss is
close to zero, even if the absolute change is small. Inspect `estimated_delta`
alongside `relative_delta` in that case.

For each `predict` call, DLE pools the held-out and adjusted current
**estimated** losses. It randomly permutes group assignments `n_resamples`
times, preserving both sample sizes, and recalculates Welch's t
statistic. The one-sided p-value uses the plus-one correction:

```python
(1 + number_of_permuted_statistics_at_least_observed) / (n_resamples + 1)
```

The smallest possible p-value is `1 / (n_resamples + 1)`, so `fit` requires
enough permutations for that value to be at or below `alpha`.

The decision requires both a relative increase above the margin and
`p_value <= alpha`. Set `random_state` to reproduce the permutation result.
With identical group distributions and independent observations, permutation
inference is exact. Studentization makes inference for equal means with
different variances an asymptotic approximation; small samples need caution.
Temporal dependence can invalidate ordinary row-wise permutations.

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
labeled reference: X, y, y_pred
              │
              ├── first 1 - fraction ──> fit learner on (y - y_pred)²
              │
              └── last fraction ───────> observed and estimated baseline loss
                                              │
                         adjust current losses by 1 + margin
                                              │
                                pool reference and adjusted losses
                                              │
                                    permute group assignments
                                              │
current: X, y_pred ──> estimated current loss ─┴─> relative_delta and p_value
                                              │
                         relative_delta > margin and p_value <= alpha?
                                               │          │
                                              yes         no
                                          degradation   no increase
```

A positive `estimated_delta` means estimated loss increased relative to the
estimated reference baseline. `degradation` means the relative increase
exceeded the chosen margin and passed the one-sided permutation test. The
p-value concerns **estimated** loss; it does
not confirm that realized performance changed. Repeated monitoring can also
produce alerts by chance, even if the reference regime remains stable.
DLE needs the learned relationship between inputs and loss to remain useful on
current data. For classification, changed probability calibration can break
that relationship. Compare estimates with realized loss as current labels
arrive.

For runnable regression, binary probability, and panel examples, see
[`dle.ipynb`](../examples/dle.ipynb).
