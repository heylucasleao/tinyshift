# Data drift

`tinyshift.drift` compares a current population with a previously fitted
reference population. It is independent of time windows:
`TemporalStabilityAnalyzer` is the component for detecting regime changes
inside a time series.

## Vector detectors

`ConDrift` and `CatDrift` accept one-dimensional lists, NumPy arrays, or pandas
Series.

```python
from tinyshift.drift import ConDrift

detector = ConDrift(
    metric="wasserstein",
    normalize=True,
    alpha=0.05,
    n_resamples=999,
    random_state=42,
).fit(reference_values)

result = detector.predict(current_values)
print(result.score, result.threshold, result.p_value, result.drift)
```

`score(current)` returns only the distribution distance and does not run
inference. `predict(current)` always runs a two-sample permutation test and
returns a `DriftResult` containing score, critical threshold, Monte Carlo
p-value, decision, and both sample sizes.

The permutation test pools reference and current observations, repeatedly
permutes their labels while preserving sample sizes, and recomputes the
statistic. Under the null hypothesis of equal distributions, it has
finite-sample validity when observations are independent and group labels are
exchangeable.

P-values use:

```python
(exceedances + 1) / (n_resamples + 1)
```

They are therefore never zero. At least `1 / alpha - 1` resamples are required
for rejection to be possible. Permutation testing with Wasserstein follows the
same two-sample principle used by
[waddR](https://doi.org/10.1093/bioinformatics/btab226).

Continuous samples use Wasserstein distance. With `normalize=True` (the
default), distance is divided by reference standard deviation. During
permutation inference, this scale is recomputed for every permuted reference.

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

analyzer = ContinuousDriftAnalyzer(ConDrift(n_resamples=999, random_state=42))
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
current batch are omitted from that result.
