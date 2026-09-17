# Data drift

`tinyshift.drift` compares a current population with a previously fitted
reference population. It is intentionally independent of time windows:
`TemporalStabilityAnalyzer` is the component for detecting regime changes
inside a time series.

## Vector detectors

`ConDrift` and `CatDrift` accept one-dimensional lists, NumPy arrays, or pandas
Series.

```python
from tinyshift.drift import ConDrift

detector = ConDrift(
    metric="wasserstein",
    threshold="permutation",
    normalize=True,
    alpha=0.05,
    random_state=42,
)
detector.fit(reference_values)

result = detector.predict(current_values)
print(result.score, result.threshold, result.p_value, result.drift)
```

`score(current)` returns only the distance. `predict(current)` returns a
`DriftResult` containing the score, threshold, p-value, classification, and
both sample sizes. With `threshold=None`, the score remains available and
`drift` is `None`. A numeric threshold can be supplied directly; manual
thresholds do not produce a p-value.

### Inference methods

- `threshold="permutation"` (default) pools reference and current observations,
  repeatedly permutes their labels while preserving sample sizes, and computes
  a finite-sample Monte Carlo p-value. Under the null, this is valid when the
  observations are independent and the group labels are exchangeable.
- `threshold="cross_conformal"` repeatedly holds out pseudo-current folds from
  the reference without replacement. It is a useful reference-only calibration
  and is cached by current sample size, but its overlapping folds make its
  p-value approximate rather than an exact conformal guarantee. Current size
  must be smaller than reference size.
- `threshold="bootstrap"` independently resamples reference and pseudo-current
  samples with replacement from the fitted empirical reference. It is retained
  as an empirical alternative and is cached by current sample size.
- `threshold=<float>` applies a fixed operational limit, and `threshold=None`
  reports only the score.

All resampling p-values use `(exceedances + 1) / (n_resamples + 1)`, so they are
never zero. At least `1 / alpha - 1` resamples are required for rejection to be
possible. Permutation testing with Wasserstein follows the same two-sample
principle used by [waddR](https://doi.org/10.1093/bioinformatics/btab226).

Continuous samples use Wasserstein distance. With `normalize=True` (the
default), distance is divided by reference standard deviation so differently
scaled features have comparable scores.

Categorical samples support:

- `metric="jensen_shannon"` (default)
- `metric="psi"`
- `metric="chebyshev"`

Reference and current categories are aligned over their union, so previously
unseen categories contribute to the distance. Missing values are rejected.

## Panel analyzers

The analyzers coordinate one independently fitted detector per ID. The input
only needs identifier and target columns; the caller decides which rows make
up the reference and current populations.

```python
from tinyshift.drift import ConDrift, ContinuousDriftAnalyzer

analyzer = ContinuousDriftAnalyzer(ConDrift(threshold="permutation", random_state=42))
analyzer.fit(reference_df, id_col="unique_id", target_col="y")

result = analyzer.predict(current_df)
```

The result has one row per current ID:

| unique_id | score | threshold | p_value | drift | reference_size | current_size |
|---|---:|---:|---:|---|---:|---:|
| A | 0.18 | 0.31 | 0.431 | false | 120 | 30 |
| B | 0.72 | 0.28 | 0.002 | true | 100 | 28 |

`CategoricalDriftAnalyzer` provides the same lifecycle for `CatDrift`. Current
IDs without a fitted reference raise an error. Reference IDs absent from a
current batch are simply omitted from that result.
