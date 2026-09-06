# Time Series Analysis (`tinyshift.series`)

Tools for analyzing observed time series: decomposition, temporal dependence,
entropy, intermittency, outliers, seasonality, spectral structure, and combined
profiling. Forecast evaluation and stabilization belong to
`tinyshift.forecasting`.

## Modules

- `decomposition`: LOWESS detrending and MSTL component extraction.
- `dependence`: permutation auto-mutual information (PAMI).
- `diagnostic`: variance-ratio and trend/seasonal significance tests.
- `entropy`: sample entropy, regularity, permutation entropy, and its derived
  theoretical predictability limit.
- `analyzers`: panel-oriented intermittency, PAMI, seasonality, and
  variance-ratio analyzers with a shared `fit()`/`summary()` convention.
- `outlier`: univariate temporal Hampel and Bollinger detectors.
- `profiler`: one combined diagnostic summary per panel series.
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
    SeasonalityAnalyzer,
    VarianceRatioAnalyzer,
)

intermittency = IntermittencyAnalyzer().fit(df).summary()
seasonality = SeasonalityAnalyzer(top_k=2).fit(df).summary()
dependence = VarianceRatioAnalyzer().fit(df).summary()
```

`IntermittencyAnalyzer` classifies demand as smooth, intermittent, erratic, or
lumpy. `SeasonalityAnalyzer` identifies spectral candidates and tests their
harmonic significance. `VarianceRatioAnalyzer` reports persistence or
mean reversion at logarithmically spaced horizons for each series.

## Diagnostics and Decomposition

- `variance_ratio`: persistence or mean reversion at one aggregation horizon.
- `trend_significance`: linear trend R² and slope p-value.
- `seasonal_strength`: strength calculated from decomposition components.
- `harmonic_significance`: harmonic-regression F-test for a candidate period.
- `detrend`: LOWESS trend and residual extraction for panel data.
- `extract_mstl_components`: conversion of an MSTL result into a tidy frame.

## Series Profiler

`SeriesProfiler` combines demand occurrence, predictability, temporal structure,
and spectral structure into one row per series:

```python
from tinyshift.series import SeriesProfiler

summary = SeriesProfiler(top_k=2).fit(
    df,
    id_col="unique_id",
    time_col="ds",
    target_col="y",
).summary()
```

The result contains `adi`, `cv2`, `zero_prop`, `interval_cv`, `class`, `foreca`,
`limit`, `trend_r2`, `trend_pvalue`, `spectral_conc`, and
`candidate_periods`. Variance-ratio analysis remains available independently
through `VarianceRatioAnalyzer`.

## Outlier Detection

- `hampel_filter`: robust rolling median/MAD detector.
- `bollinger_bands`: rolling volatility-envelope detector.

Both return a boolean `pandas.Series` and preserve a supplied Series index.

For forecast metrics, stabilization, or decomposed forecasting wrappers, see
[`tinyshift.forecasting`](../forecasting/README.md).
