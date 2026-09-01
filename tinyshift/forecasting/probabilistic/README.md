# Two-Stage Forecasting (`forecasting.probabilistic`)

The `forecasting.probabilistic` package builds parametric predictive distributions around
MLForecast point forecasts. It is the internal implementation behind
`TwoStageForecasterWrapper`, the distribution families, panel forecast facades,
evaluation helpers, and Newsvendor decisions exported from
`tinyshift.forecasting`.

## Public API

| Object | Responsibility |
|---|---|
| `TwoStageForecasterWrapper` | Fit the point forecaster and calibrate hierarchical dispersion |
| `NegativeBinomialFamily` | Model non-negative integer counts |
| `GammaFamily` | Model strictly positive continuous targets |
| `PanelPredictiveForecast` | Expose CDFs, quantiles and central intervals on the forecast panel |
| `DiscretePanelPredictiveForecast` | Additionally expose probability masses and integer quantiles |
| `NewsvendorOptimizer` | Convert predictive distributions into inventory decisions |
| `FirstStageForecasterEvaluator` | Evaluate the conditional-mean forecasting stage |
| `TwoStageForecasterEvaluator` | Evaluate the complete probabilistic forecast |

Users should normally import these objects from `tinyshift.forecasting`:

```python
from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression
from tinyshift.forecasting import TwoStageForecasterWrapper

point_forecaster = MLForecast(
    models=[LinearRegression()],
    freq="D",
    lags=[1, 7],
)

model = TwoStageForecasterWrapper(point_forecaster).fit(
    train_df,
    h=14,
    n_windows=5,
    step_size=14,
)

forecast = model.predict_distribution(h=14, X_df=future_exog)
median = forecast.ppf(0.5)
interval = forecast.interval(coverage=0.9)
probabilities = forecast.cdf(values)
masses = forecast.pmf([0, 1, 2])
```

The default family is Negative Binomial, so the returned forecast is discrete.
Pass `distribution=GammaFamily()` for a continuous positive target; Gamma
forecasts intentionally do not expose `pmf`.

## Internal modules

| Module | Responsibility |
|---|---|
| `wrapper.py` | Estimator lifecycle, temporal calibration and forecast construction |
| `family.py` | Target support, likelihood optimization and distribution factories |
| `calibration.py` | Hierarchical shrinkage state and fallback policy |
| `distribution.py` | Row-aligned parametric CDF, PPF, interval, sampling and PMF mathematics |
| `forecast.py` | DataFrame facade that keeps distribution outputs aligned with panel rows |
| `decision.py` | Newsvendor critical-ratio and marginal-benefit policies |
| `eval.py` | Point and probabilistic forecast metrics |
| `__init__.py` | Public package exports |

Dependencies flow toward the smaller statistical components. `wrapper.py`
coordinates fitting and prediction, while families, distributions, forecast
facades, decisions, and evaluators remain independent from the estimator
lifecycle.

## Fitting flow

`fit` validates the selected family and obtains rolling temporal
cross-validation predictions from MLForecast. Dispersion is fitted by bounded
maximum likelihood globally, per series, and per series×horizon. Estimates are
shrunk in `log(dispersion)` toward their parent using weights inferred from the
likelihood curvature and empirical between-group variance. No regularization
constant is required from the user. The global fit is retained as the fallback
for series not seen during calibration. MLForecast is then fitted on all rows.

Fitted layers are stored in `calibration_.dispersion`: `global`,
`global_horizon`, `series`, and `series_horizon`. Prediction resolves known
series through `series_horizon -> series -> global`. For an unknown series it
uses `global_horizon -> global`. The corresponding between-group variances are
available in `calibration_.tau2`.

```text
training panel
      |
      v
rolling temporal cross-validation
      |
      v
out-of-fold conditional means by series
      |
      +--> global family-specific dispersion
      +--> shrunk global-horizon dispersion
      +--> shrunk per-series dispersion
      +--> shrunk per-series×horizon dispersion
      `--> global fallback
                    |
                    v
full point-forecaster fit
                    |
                    v
point means + aligned dispersions
                    |
                    v
panel-aligned predictive forecast
```

## Distribution semantics

Every predictive-distribution row corresponds positionally to one row returned
by `forecast.to_frame()`. A scalar input is evaluated for every row, a
one-dimensional input defines a common grid, and a two-dimensional array with
`len(forecast)` rows supplies row-wise values.

Negative Binomial distributions use the conditional mean and calibrated size
parameter. Gamma distributions use the conditional mean and calibrated shape.
Both expose `cdf`, `ppf`, and `interval` internally. Discrete
distributions additionally define `pmf(k) = cdf(k) - cdf(k - 1)`.

The forecast facade returns DataFrames and names quantile columns as percentages,
for example `lambda_t-q-50` and `lambda_t-q-90`. Distribution implementation
classes remain internal and are intentionally excluded from `forecasting.probabilistic.__all__`.

## Decision flow

`NewsvendorOptimizer.optimize` evaluates the inverse CDF at the critical ratio
`underage_cost / (underage_cost + overage_cost)`. Costs may be numeric scalars,
DataFrame columns, mappings by series, or mappings by `(series, timestamp)`.

For discrete distributions, `marginal_benefit` computes the expected net value
of each additional inventory unit from the predictive CDF. Decision outputs
preserve the original forecast row order.

## Alignment and extension rules

- Never reorder a forecast frame independently from its predictive distribution.
- Add distribution mathematics to `distribution.py`, not to the DataFrame facade.
- Add target validation, likelihoods, and distribution construction to a
  `DistributionFamily` implementation.
- Keep `wrapper.py` focused on orchestration and MLForecast integration.
- Keep column formatting in `forecast.py` and decision policies in `decision.py`.
- Validate all distribution parameters and forecast means as finite and positive.
- Preserve the series and global fallbacks when extending dispersion calibration.

Tests for this package live in `tinyshift/tests/test_tsf.py`. Run them with:

```bash
pytest -q tinyshift/tests/test_tsf.py
```
