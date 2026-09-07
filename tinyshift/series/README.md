# Time Series Analysis (`tinyshift.series`)

Tools for analyzing observed time series: temporal dependence, entropy,
intermittency, seasonality, spectral structure, and combined profiling. Forecast
decomposition, evaluation, and stabilization belong to
`tinyshift.forecasting`.

## Modules

- `dependence`: permutation auto-mutual information (PAMI).
- `diagnostic`: variance-ratio and trend/seasonal significance tests.
- `entropy`: sample entropy, regularity, permutation entropy, and its derived
  theoretical predictability limit.
- `analyzers`: panel-oriented intermittency, PAMI, predictability, seasonality,
  trend, and variance-ratio analyzers with a shared `fit()`/`summary()` convention.
  See the [analyzer reference](analyzers/README.md).
- `spectral`: shared spectrum preparation, ForeCA, and spectral concentration.

## Spectral, Entropy, and Dependence Metrics

- `foreca`: spectral-entropy forecastability index from 0 to 1.
- `spectral_concentration`: normalized concentration of power across frequencies.
- `sample_entropy`: magnitude-based irregularity and complexity.
- `regularity_index`: regularity score derived from sample entropy.
- `permutation_entropy`: ordinal-pattern complexity, optionally normalized.
- `theoretical_limit`: ordinal predictability ceiling derived from normalized
  permutation entropy.
- `permutation_auto_mutual_information`: non-linear dependence between ordinal
  patterns separated by a lag.
- `PAMIAnalyzer`: finds every local minimum of each panel series' PAMI curve.
- `create_pami_lags`: converts minima into DTL/DMSTL lag dictionaries.

## Forecastability Dimensions

Forecastability is best understood as a multidimensional diagnostic rather than
as a single score. The tools in this module inspect complementary sources of
predictable structure:

$$
\mathrm{Forecastability}
\;\leftarrow\;
\begin{cases}
\mathrm{regularity} & \mathrm{sample\ entropy\ and\ regularity\ index} \\
\mathrm{periodicity} & \mathrm{seasonality\ analyzer\ and\ harmonic\ tests} \\
\mathrm{spectral\ structure} & \mathrm{ForeCA\ and\ spectral\ concentration} \\
\mathrm{persistence} & \mathrm{variance\ ratio\ and\ temporal\ dependence} \\
\mathrm{intermittency} & \mathrm{intermittency\ analyzer} \\
\mathrm{ordinal\ predictability} & \mathrm{permutation\ entropy\ and\ theoretical\ limit}
\end{cases}
$$

These dimensions should be interpreted together. For example, a series may have
strong periodicity but weak persistence, or a concentrated spectrum while still
being highly intermittent. The resulting profile is useful for choosing a
forecasting strategy and model assumptions, but it is not an additive
forecastability formula.

```python
from tinyshift.series import (
    foreca,
    PAMIAnalyzer,
    create_pami_lags,
    permutation_auto_mutual_information,
    permutation_entropy,
    spectral_concentration,
    theoretical_limit,
)

omega = foreca(values)
concentration = spectral_concentration(values)
complexity = permutation_entropy(values)
limit = theoretical_limit(values)
pami = permutation_auto_mutual_information(values, tau=7)
pami = PAMIAnalyzer(max_tau=30).fit(df)
minima = pami.summary()
lags = pami.lags(mode="short", short=2, fallback=1)
```

## Intermittency, Seasonality, and Variance Ratio

```python
from tinyshift.series import (
    IntermittencyAnalyzer,
    PredictabilityAnalyzer,
    SeasonalityAnalyzer,
    TrendAnalyzer,
    VarianceRatioAnalyzer,
)

intermittency = IntermittencyAnalyzer().fit(df).summary()
predictability = PredictabilityAnalyzer().fit(df).summary()
seasonality = SeasonalityAnalyzer(top_k=2).fit(df).summary()
trend = TrendAnalyzer().fit(df).summary()
dependence = VarianceRatioAnalyzer().fit(df).summary()
```

`IntermittencyAnalyzer` classifies demand as smooth, intermittent, erratic, or
lumpy. `SeasonalityAnalyzer` identifies spectral candidates and tests their
harmonic significance. `VarianceRatioAnalyzer` reports persistence or
mean reversion at logarithmically spaced horizons for each series.

## Diagnostics and Decomposition

- `variance_ratio`: persistence or mean reversion at one aggregation horizon.
- `trend_significance`: linear trend R² and slope p-value.
- `harmonic_significance`: harmonic-regression F-test for a candidate period.

`seasonal_strength`, `detrend`, and `extract_mstl_components` are internal
forecasting helpers. They live in `tinyshift.forecasting.dmstl.utils` and
`tinyshift.forecasting.dtl.utils`, respectively, rather than in the public
`tinyshift.series` API.

## Combining Analyzer Summaries

Analyzer summaries can be combined explicitly with validated one-to-one merges:

```python
analyzers = [
    IntermittencyAnalyzer(),
    PredictabilityAnalyzer(),
    TrendAnalyzer(),
    SeasonalityAnalyzer(top_k=2),
]

summaries = [analyzer.fit(df).summary() for analyzer in analyzers]
summary = summaries[0]
for section in summaries[1:]:
    summary = summary.merge(section, on="unique_id", validate="one_to_one")
```

The result contains `adi`, `cv2`, `zero_proportion`, `interval_cv`, `classification`, `foreca`,
`limit`, `spectral_concentration`, the linear-trend diagnostics, and candidate
and significant seasonal periods. Variance-ratio analysis remains available independently
through `VarianceRatioAnalyzer`.

For forecast metrics, stabilization, or decomposed forecasting wrappers, see
[`tinyshift.forecasting`](../forecasting/README.md).
