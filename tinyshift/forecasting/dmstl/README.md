# Decomposed Multi-Seasonal Forecasting (`forecasting.dmstl`)

The `forecasting.dmstl` package forecasts panels by decomposing each series into
trend, one or more seasonal components, and residual dynamics with MSTL. Trend
and seasonal components are forecast with StatsForecast; residuals are modeled
with MLForecast. The public `DMSTLWrapper` facade selects a local or global
residual strategy.

## Public API

Only `DMSTLWrapper` is public. Users should import it from
`tinyshift.forecasting`:

```python
from mlforecast import MLForecast
from sklearn.ensemble import RandomForestRegressor
from tinyshift.forecasting import DMSTLWrapper


def residual_model(nlags, freq):
    return MLForecast(
        models=[RandomForestRegressor(n_estimators=100, random_state=0)],
        lags=nlags,
        freq=freq,
    )


model = DMSTLWrapper(
    mode="global",
    residual_model_callable=residual_model,
    freq="D",
    season_length=[7, 30],
    nlags="auto",
    pami_params={"max_tau": 48, "m": 3, "delay": 1},
).fit(train_df)

forecast = model.predict(h=14, X_df=future_exog, level=[80, 95])
```

`season_length="auto"` may be used to detect periods independently for each
series. The detector first finds spectral candidates and then keeps only periods
whose harmonic significance test passes. Explicit integers, lists, or mappings
by `unique_id` are supported when seasonality is known beforehand.

| Mode | Residual model | Configuration |
|---|---|---|
| `local` | One MLForecast instance per series | Residual factories and lags may be configured by `unique_id` |
| `global` | One MLForecast instance for the complete residual panel | One shared factory receives the union of all resolved lags |

Trend and seasonal values remain series-specific in both modes. Global mode
shares only the residual learner.

## Internal modules

| Module | Responsibility |
|---|---|
| `wrapper.py` | Public facade, strategy selection and fitted-attribute forwarding |
| `base.py` | Period resolution, MSTL decomposition, component fitting and recombination |
| `local_.py` | Per-series residual MLForecast fitting and prediction |
| `global_.py` | Shared panel residual MLForecast fitting and prediction |
| `utils.py` | MSTL component extraction and post-decomposition seasonal strength |
| `__init__.py` | Public export of `DMSTLWrapper` |

`BaseDMSTL` owns the statistical workflow. Concrete strategies implement only
how residual frames are fitted and predicted. The facade creates the selected
delegate and mirrors its fitted attributes.

## Fitting flow

Each panel series is sorted by time and optionally transformed with `log1p`.
Seasonal periods are resolved from configuration or automatic detection, then
validated against the available history. MSTL splits the target into trend,
seasonal, and residual components.

```text
long-format panel
       |
       v
sort each series by time
       |
       +--> optional log1p
       v
resolve and validate seasonal periods
       |
       v
MSTL decomposition
       |
       +--> trend ----------> StatsForecast trend model
       +--> seasonal periods -> StatsForecast model per period
       `--> residual --------> explicit or PAMI-selected lags
                                      |
                                      +--> local MLForecast per series
                                      `--> global MLForecast on residual panel
```

Series using the same trend factory are batched into one StatsForecast panel
fit. Seasonal components are batched by `(period, factory)`. Batching reduces
fit calls without pooling component values or fitted parameters across series.

## Seasonal-period semantics

`season_length` accepts an integer, a list of integers, a mapping by
`unique_id`, or `"auto"`. Every period must be an integer greater than one.
Automatic detection uses `SeasonalityAnalyzer` configured with
`seasonal_detection_params`. The DMSTL workflow consumes the analyzer's
`significant_periods`, not every spectral `candidate_periods`. If no candidate
passes the harmonic test, the configured `fallback` is used when available;
otherwise fitting raises an error.

A series must contain enough history for MSTL. One seasonal period requires at
least twice that period; multiple periods require at least twice their sum.
When automatic detection cannot find a period, configure an explicit period or
provide a suitable detection fallback.

## Prediction and recombination

The residual strategy first predicts the residual panel. DMSTL then adds the
series-specific trend forecast and every seasonal forecast to each residual
point or interval column. With `log_transform=True`, `expm1` is applied after
all components are recombined.

Optional horizontal stabilization uses `stabilization_method="hpi"` or
`"hfi"` with a positive `w_s`. When training contains exogenous columns,
`X_df` is required for prediction. Local mode filters future features by
series; global mode sends the complete panel to its shared residual model.

## Configuration rules

- `freq` must be explicitly configured.
- Trend and seasonal model factories may be shared or mapped by `unique_id`.
- A seasonal factory receives the corresponding integer period.
- Local residual factories may be shared or mapped by `unique_id`; global mode
  requires one callable.
- `nlags="auto"` runs PAMI per series. Global mode uses the sorted union of all
  selected residual lags.
- Prediction intervals are produced by the residual MLForecast model and are
  recombined with the same trend and seasonal components as point forecasts.

## Extension rules

- Keep period validation, decomposition and recombination in `base.py`.
- Keep strategy-specific residual behavior in `local_.py` and `global_.py`.
- Preserve one decomposition and component forecast per series.
- Batch StatsForecast models only when their resolved factory and, for seasonal
  models, period match.
- Apply inverse transformations and stabilization only after recombination.
- Preserve identifier and timestamp columns when joining component forecasts.
- Keep MSTL-specific helpers in `utils.py`; `seasonal_strength` is a diagnostic
  of an already-fitted decomposition and is not used to select periods.

Tests live in `tinyshift/tests/test_dmstl.py`:

```bash
pytest -q tinyshift/tests/test_dmstl.py
```
