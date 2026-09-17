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
    alpha=0.05,
    n_resamples=999,
    random_state=42,
).fit(reference_values)

result = detector.predict(current_values)
print(result.score, result.threshold, result.p_value, result.drift)
```

`predict(current)` runs a two-sample permutation test and returns a
`DriftResult` containing the internal distance, critical threshold, Monte
Carlo p-value, decision, and both sample sizes. Users should base monitoring
decisions on `p_value` and `drift`; the distance is retained for analyzer
reporting and diagnostics.

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

## Score interpretation

Both detectors return unit-independent scores, but their ranges are different.
The permutation p-value, rather than an arbitrary score cutoff, determines the
drift decision.

### Continuous score

`ConDrift` uses Wasserstein distance divided by the reference standard
deviation:

```text
0 ------------------------------------------------------------> ∞
identical distributions                     increasing change
```

Examples:

- `0.0`: the empirical distributions are identical;
- `0.5`: the Wasserstein distance is half the reference standard deviation;
- `1.0`: the Wasserstein distance equals one reference standard deviation;
- `2.0`: the Wasserstein distance equals two reference standard deviations.

The score is not bounded above. During permutation inference, the standard
deviation is recomputed from every permuted reference group.

### Categorical score

`CatDrift` uses Jensen–Shannon distance with logarithm base 2:

```text
0 ------------------------------------------------------------> 1
identical distributions                  disjoint distributions
```

Examples:

- `0.0`: the categorical distributions are identical;
- values near `0.0`: category proportions are similar;
- values near `1.0`: the distributions have little overlap;
- `1.0`: their supports are completely disjoint.

SciPy's `jensenshannon` returns the square root of the Jensen–Shannon
divergence. Using `base=2` bounds that distance to `[0, 1]`. Reference and
current categories are aligned over their union, so previously unseen
categories contribute to the score. Missing values are rejected.

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

```
                   BaseDrift
                       │
                permutation test
                       │
          ┌────────────┴────────────┐
          │                         │
       ConDrift                  CatDrift
          │                         │
   Wasserstein               Jensen-Shannon
          │                         │
          └────────────┬────────────┘
                       │
               observed score
                       │
                Monte Carlo H0
                       │
                    p-value
                       │
                p <= alpha ?
                  /          \
                yes           no
              drift        no drift
```