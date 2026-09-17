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
    threshold="bootstrap",
    normalize=True,
    alpha=0.05,
    random_state=42,
)
detector.fit(reference_values)

result = detector.predict(current_values)
print(result.score, result.threshold, result.drift)
```

`score(current)` returns only the distance. `predict(current)` returns a
`DriftResult` containing the score, threshold, classification, and both sample
sizes. With `threshold=None`, the score remains available and `drift` is
`None`. A numeric threshold can be supplied directly.

Bootstrap thresholds are calibrated under the fitted empirical reference and
cached by current sample size. This matters because the expected sampling
variation changes with batch size.

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

analyzer = ContinuousDriftAnalyzer(ConDrift(threshold="bootstrap", random_state=42))
analyzer.fit(reference_df, id_col="unique_id", target_col="y")

result = analyzer.predict(current_df)
```

The result has one row per current ID:

| unique_id | score | threshold | drift | reference_size | current_size |
|---|---:|---:|---|---:|---:|
| A | 0.18 | 0.31 | false | 120 | 30 |
| B | 0.72 | 0.28 | true | 100 | 28 |

`CategoricalDriftAnalyzer` provides the same lifecycle for `CatDrift`. Current
IDs without a fitted reference raise an error. Reference IDs absent from a
current batch are simply omitted from that result.
