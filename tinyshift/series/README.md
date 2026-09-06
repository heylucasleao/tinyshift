# Time Series Analysis (`tinyshift.series`)

Tools for analyzing observed time series: decomposition, temporal dependence,
entropy, intermittency, outliers, seasonality, spectral structure, and combined
profiling. Forecast evaluation and stabilization belong to
`tinyshift.forecasting`.

## Modules

- `decomposition`: LOWESS detrending and MSTL component extraction.
- `dependence`: permutation auto-mutual information (PAMI) and lag selection.
- `diagnostic`: Hurst exponent and trend/seasonal significance.
- `entropy`: sample entropy, regularity, permutation entropy, and its derived
  theoretical predictability limit.
- `intermittency`: ADI, CV², zero proportion, interval variability, and demand
  classification.
- `outlier`: univariate temporal Hampel and Bollinger detectors.
- `profiler`: one combined diagnostic summary per panel series.
- `seasonality`: FFT-based candidate-period detection.
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
- `select_pami_lag`: selects a lag from the first local minimum of the PAMI curve.

```python
from tinyshift.series import (
    foreca,
    permutation_auto_mutual_information,
    permutation_entropy,
    select_pami_lag,
    spectral_concentration,
    theoretical_limit,
)

omega = foreca(values)
concentration = spectral_concentration(values)
complexity = permutation_entropy(values)
limit = theoretical_limit(values)
pami = permutation_auto_mutual_information(values, tau=7)
lags, selected_pami, curve = select_pami_lag(values, max_tau=30, fallback=1)
```

## Intermittency and Seasonality

```python
from tinyshift.series import IntermittencyAnalyzer, SeasonalPeriodDetector

intermittency = IntermittencyAnalyzer().fit(df).profile()
seasonality = SeasonalPeriodDetector(top_k=2).fit(df).profile()
```

`IntermittencyAnalyzer` classifies demand as smooth, intermittent, erratic, or
lumpy. `SeasonalPeriodDetector` identifies candidate seasonal periods from
significant spectral peaks.

## Diagnostics and Decomposition

- `hurst_exponent`: long-memory estimate and random-walk p-value.
- `trend_significance`: linear trend R² and slope p-value.
- `seasonal_significance`: seasonal strength and harmonic-regression F-test.
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
`limit`, `hurst`, `trend_r2`, `trend_pvalue`, `spectral_conc`, and
`candidate_periods`. Each series needs at least 30 observations because the
profile includes the Hurst exponent.

## Outlier Detection

- `hampel_filter`: robust rolling median/MAD detector.
- `bollinger_bands`: rolling volatility-envelope detector.

Both return a boolean `pandas.Series` and preserve a supplied Series index.

For forecast metrics, stabilization, or decomposed forecasting wrappers, see
[`tinyshift.forecasting`](../forecasting/README.md).
