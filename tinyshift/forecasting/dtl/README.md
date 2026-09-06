# Decomposed Trend Forecasting (`forecasting.dtl`)

The `forecasting.dtl` package forecasts non-seasonal panels by separating a
smooth LOWESS trend from the remaining residual dynamics. Trend components are
forecast with StatsForecast and residuals with MLForecast. The public
`DTLWrapper` facade selects a local or global residual strategy.

## Public API

Only `DTLWrapper` is public. Users should import it from `tinyshift.forecasting`:

```python
from mlforecast import MLForecast
from sklearn.ensemble import RandomForestRegressor
from tinyshift.forecasting import DTLWrapper


def residual_model(nlags, freq):
    return MLForecast(
        models=[RandomForestRegressor(n_estimators=100, random_state=0)],
        lags=nlags,
        freq=freq,
    )


model = DTLWrapper(
    mode="global",
    residual_model_callable=residual_model,
    freq="D",
    trend_frac=0.2,
    robust=True,
    nlags="auto",
    pami_params={"max_tau": 48, "m": 3, "delay": 1},
).fit(train_df)

forecast = model.predict(h=14, X_df=future_exog, level=[80, 95])
```

| Mode | Residual model | Configuration |
|---|---|---|
| `local` | One MLForecast instance per series | Residual factories and lags may be configured by `unique_id` |
| `global` | One MLForecast instance for the complete residual panel | One shared factory receives the union of all resolved lags |

The trend is always estimated and forecast per series. Global mode shares only
the residual learner across the panel.

## Internal modules

| Module | Responsibility |
|---|---|
| `wrapper.py` | Public facade, strategy selection and fitted-attribute forwarding |
| `base.py` | LOWESS decomposition, trend fitting, lag resolution and forecast recombination |
| `utils.py` | Internal LOWESS detrending helper for panel data |
| `local_.py` | Per-series residual MLForecast fitting and prediction |
| `global_.py` | Shared panel residual MLForecast fitting and prediction |
| `__init__.py` | Public export of `DTLWrapper` |

`BaseDTL` owns the common workflow. The local and global implementations only
specialize residual fitting and prediction; the public wrapper constructs the
selected delegate and exposes its fitted attributes.

## Fitting flow

Each series is sorted by timestamp before decomposition. Optional `log1p`
transformation is applied first. LOWESS then produces a trend and a detrended
residual series. Residual lags are configured explicitly or selected per series
with PAMI.

```text
long-format panel
       |
       v
sort each series by time
       |
       +--> optional log1p
       v
LOWESS decomposition
       |
       +--> trend ----> per-series StatsForecast parameters
       |
       `--> residual -> explicit or PAMI-selected lags
                              |
                              +--> local MLForecast per series
                              `--> global MLForecast on residual panel
```

Series resolving to the same trend-model factory are batched into one
StatsForecast panel fit. The fitted object may therefore be shared, but
StatsForecast still maintains separate fitted parameters and forecasts for each
`unique_id`.

## Prediction and recombination

The selected residual strategy first produces an MLForecast panel. DTL then
predicts each trend component and adds it to every residual point or interval
column for that series. With `log_transform=True`, `expm1` is applied after
recombination.

Optional horizontal stabilization can be requested with
`stabilization_method="hpi"` or `"hfi"` and a positive `w_s`. If exogenous
features were present during fitting, `X_df` is required during prediction.
Local mode filters it by series; global mode passes the complete frame to the
shared residual model.

## Configuration rules

- `freq` must be explicitly configured.
- `trend_model_callable` may be shared or mapped by `unique_id`.
- Local residual factories may be shared or mapped by `unique_id`; global mode
  requires one callable.
- An integer `nlags=n` expands to lags `1..n`; a list is used directly.
- `nlags="auto"` runs PAMI independently for every series. Global mode uses the
  sorted union of the selected lags.
- DTL is intended for non-seasonal series. Use DMSTL when explicit seasonal
  components need to be decomposed and forecast.

## Extension rules

- Keep decomposition, trend handling and recombination in `base.py`.
- Keep strategy-specific residual behavior in `local_.py` and `global_.py`.
- Preserve panel row identifiers and timestamps throughout residual prediction.
- Do not share trend values across series, even when their StatsForecast fit is
  batched.
- Apply inverse transformations and stabilization only after component
  recombination.
- Keep the panel LOWESS helper in `utils.py` rather than exposing it through the
  public series API.

Tests live in `tinyshift/tests/test_dtl.py`:

```bash
pytest -q tinyshift/tests/test_dtl.py
```
