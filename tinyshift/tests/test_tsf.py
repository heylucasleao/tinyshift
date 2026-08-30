import joblib
import numpy as np
import pandas as pd
import pytest
from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression

import tinyshift.modelling.tsf.family as tsf_family_module
import tinyshift.modelling.tsf.wrapper as tsf_wrapper_module
from tinyshift.modelling import (
    FirstStageForecasterEvaluator,
    GammaFamily,
    GammaPredictiveDistribution,
    NegativeBinomialFamily,
    NegativeBinomialPredictiveDistribution,
    TwoStageForecasterEvaluator,
    TwoStageForecasterWrapper,
)


@pytest.fixture
def sample_train_data():
    """Generates a pure pandas DataFrame for testing without extra exogenous columns."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
    data = []
    for uid in ["A", "B"]:
        y_vals = np.random.poisson(lam=3, size=200)
        for ds, y in zip(dates, y_vals):
            data.append({"unique_id": uid, "ds": ds, "y": float(y)})
    return pd.DataFrame(data)


@pytest.fixture
def sample_train_data_with_costs(sample_train_data):
    """Adds cost columns separately for optimization tests."""
    df = sample_train_data.copy()
    df["cu"] = 10.0
    df["co"] = 2.0
    return df


@pytest.fixture
def sample_continuous_data():
    dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
    data = []
    for offset, uid in enumerate(["A", "B"]):
        values = 5.0 + offset + np.sin(np.arange(200) / 7.0)
        for ds, y in zip(dates, values):
            data.append({"unique_id": uid, "ds": ds, "y": y})
    return pd.DataFrame(data)


def test_init():
    base_model = LinearRegression()
    fcst = MLForecast(models=[base_model], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst)
    assert wrapper.fcst is fcst


def test_model_property_unfitted():
    base_model = LinearRegression()
    fcst = MLForecast(models=[base_model], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst)
    with pytest.raises(
        ValueError, match="The MLForecast object has not been fitted yet."
    ):
        _ = wrapper.model


def test_model_property_multiple_models(sample_train_data):
    fcst = MLForecast(
        models=[LinearRegression(), LinearRegression()], freq="D", lags=[1]
    )
    wrapper = TwoStageForecasterWrapper(fcst=fcst)
    with pytest.raises(
        ValueError, match="TwoStageForecasterWrapper supports exactly 1 model"
    ):
        wrapper.fit(sample_train_data)
        _ = wrapper.model


def test_nbinom_log_likelihood():
    wrapper = TwoStageForecasterWrapper(fcst=None)
    y = np.array([1, 2, 0, 4])
    lambda_t = np.array([1.5, 1.8, 0.5, 3.5])
    nll = wrapper._nbinom_log_likelihood([2.5], y, lambda_t)
    assert isinstance(nll, float)
    assert not np.isnan(nll)

    nll_invalid = wrapper._nbinom_log_likelihood([0.0], y, lambda_t)
    assert nll_invalid == 1e10


def test_estimate_r():
    wrapper = TwoStageForecasterWrapper(fcst=None)
    y_obs = np.array([0, 1, 2, 3, 5, 1, 0, 4])
    lambdas = np.array([1.0, 1.2, 1.5, 2.0, 2.5, 1.8, 1.0, 3.0])
    r_est = wrapper._estimate_r(y_obs, lambdas)
    assert isinstance(r_est, float)
    assert 1e-3 <= r_est <= 50.0


def test_estimate_r_raises_when_optimizer_fails(monkeypatch):
    class FailedResult:
        success = False
        message = "did not converge"
        x = 1.0
        fun = np.inf

    monkeypatch.setattr(
        tsf_wrapper_module,
        "minimize_scalar",
        lambda *args, **kwargs: FailedResult(),
    )
    wrapper = TwoStageForecasterWrapper(fcst=None)

    with pytest.raises(RuntimeError, match="Dispersion optimization failed"):
        wrapper._estimate_r(np.array([1.0]), np.array([1.0]))


def test_estimate_r_rejects_non_finite_lambdas():
    wrapper = TwoStageForecasterWrapper(fcst=None)
    with pytest.raises(ValueError, match="lambda values must be finite"):
        wrapper._estimate_r(np.array([1.0]), np.array([np.nan]))


def test_compute_time_decay_weights(sample_train_data):
    wrapper = TwoStageForecasterWrapper(fcst=None)
    weights = wrapper._compute_time_decay_weights(
        sample_train_data, time_col="ds", gamma=0.5
    )
    assert isinstance(weights, np.ndarray)
    assert len(weights) == len(sample_train_data)
    assert np.all(weights > 0)
    assert np.all(weights <= 1.0)


def test_compute_quantile():
    df = pd.DataFrame({"r_dispersion": [2.0, 5.0], "lambda_t": [3.0, 1.5]})
    quantiles = TwoStageForecasterWrapper._compute_quantile(df, target_q=0.95)
    assert isinstance(quantiles, np.ndarray)
    assert len(quantiles) == 2
    assert np.issubdtype(quantiles.dtype, np.integer)


@pytest.mark.parametrize("quantile", [-0.1, 0.0, 1.0, 1.1])
def test_compute_quantile_rejects_invalid_probability(quantile):
    df = pd.DataFrame({"r_dispersion": [2.0], "lambda_t": [3.0]})
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        TwoStageForecasterWrapper._compute_quantile(df, target_q=quantile)


def test_compute_critical_quantile():
    wrapper = TwoStageForecasterWrapper(fcst=None)
    cu = np.array([10.0, 5.0, 0.0])
    co = np.array([2.0, 5.0, 0.0])
    q_star = wrapper._compute_critical_quantile(cu, co)

    assert isinstance(q_star, np.ndarray)
    assert q_star[0] == pytest.approx(10.0 / 12.0)
    assert q_star[1] == pytest.approx(5.0 / 10.0)
    assert q_star[2] == 0.5


def test_compute_critical_quantile_rejects_negative_costs():
    wrapper = TwoStageForecasterWrapper(fcst=None)
    with pytest.raises(ValueError, match="non-negative"):
        wrapper._compute_critical_quantile(cu=np.array([-1.0]), co=np.array([2.0]))


def test_extract_cost_array(sample_train_data):
    wrapper = TwoStageForecasterWrapper(fcst=None)
    n_rows = len(sample_train_data)

    arr_scalar = wrapper._extract_cost_array(
        sample_train_data, 5.0, "unique_id", "ds", n_rows
    )
    assert np.all(arr_scalar == 5.0)

    sample_train_data["cu"] = 10.0
    arr_col = wrapper._extract_cost_array(
        sample_train_data, "cu", "unique_id", "ds", n_rows
    )
    assert np.array_equal(arr_col, sample_train_data["cu"].to_numpy(dtype=float))

    dict_map = {"A": 15.0, "B": 25.0}
    arr_dict = wrapper._extract_cost_array(
        sample_train_data, dict_map, "unique_id", "ds", n_rows
    )
    assert len(arr_dict) == n_rows

    tuple_dict = {("A", sample_train_data["ds"].iloc[0]): 99.0}
    arr_tuple = wrapper._extract_cost_array(
        sample_train_data, tuple_dict, "unique_id", "ds", n_rows
    )
    assert np.isnan(arr_tuple[1])

    with pytest.raises(ValueError, match="Cost dictionary cannot be empty."):
        wrapper._extract_cost_array(sample_train_data, {}, "unique_id", "ds", n_rows)

    with pytest.raises(TypeError, match="Cost input must be a column name"):
        wrapper._extract_cost_array(
            sample_train_data, [1, 2, 3], "unique_id", "ds", n_rows
        )


@pytest.mark.parametrize(
    ("invalid_y", "message"),
    [
        (-1.0, "non-negative"),
        (1.5, "integer counts"),
        (np.nan, "finite"),
        (np.inf, "finite"),
        ("invalid", "numeric counts"),
    ],
)
def test_fit_rejects_invalid_target(sample_train_data, invalid_y, message):
    df = sample_train_data.copy()
    if isinstance(invalid_y, str):
        df["y"] = df["y"].astype(object)
    df.loc[df.index[0], "y"] = invalid_y
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst)

    with pytest.raises(ValueError, match=message):
        wrapper.fit(df)


@pytest.mark.parametrize(
    ("df", "message"),
    [
        (pd.DataFrame(columns=["unique_id", "ds", "y"]), "cannot be empty"),
        (
            pd.DataFrame({"unique_id": ["A"], "ds": [pd.Timestamp("2023-01-01")]}),
            "was not found",
        ),
    ],
)
def test_fit_rejects_missing_or_empty_target(df, message):
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst)

    with pytest.raises(ValueError, match=message):
        wrapper.fit(df)


def test_fit_and_predict_without_x_df(sample_train_data):
    """Tests fitting and predicting without passing X_df (relies on mlforecast future generation)."""
    base_model = LinearRegression()
    fcst = MLForecast(models=[base_model], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst)
    wrapper.fit(sample_train_data)

    pred_df = wrapper.predict(h=3, X_df=None, quantiles=[0.5, 0.95])
    assert isinstance(pred_df, pd.DataFrame)
    assert "lambda_t" in pred_df.columns
    assert "q_50" in pred_df.columns
    assert len(pred_df) == 6  # 2 IDs * 3 horizon


def test_default_family_returns_negative_binomial_distribution(sample_train_data):
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst).fit(sample_train_data)

    frame, distribution = wrapper.predict_distribution(h=2)

    assert isinstance(distribution, NegativeBinomialPredictiveDistribution)
    assert len(distribution) == len(frame)
    assert distribution.cdf(frame["lambda_t"].to_numpy()).shape == (len(frame),)
    assert distribution.pmf(np.arange(3)).shape == (len(frame), 3)


def test_gamma_family_supports_continuous_targets(sample_continuous_data):
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1, 7])
    wrapper = TwoStageForecasterWrapper(fcst=fcst, distribution=GammaFamily()).fit(
        sample_continuous_data, h=7, n_windows=3
    )

    frame = wrapper.predict(h=2, quantiles=[0.1, 0.5, 0.9])
    distribution_frame, distribution = wrapper.predict_distribution(h=2)

    assert "shape_dispersion" in frame
    assert np.issubdtype(frame["q_50"].dtype, np.floating)
    assert isinstance(distribution, GammaPredictiveDistribution)
    assert distribution.interval(0.9).shape == (len(distribution_frame), 2)
    assert distribution.sample(3, random_state=42).shape == (
        len(distribution_frame),
        3,
    )


def test_gamma_family_rejects_zero_and_has_no_pmf(sample_continuous_data):
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst, distribution=GammaFamily())
    invalid = sample_continuous_data.copy()
    invalid.loc[invalid.index[0], "y"] = 0.0

    with pytest.raises(ValueError, match="strictly positive"):
        wrapper.fit(invalid)

    wrapper.fit(sample_continuous_data)
    with pytest.raises(TypeError, match="only for discrete"):
        wrapper.pmf(h=1)


def test_optimize_uses_continuous_distribution_ppf(sample_continuous_data):
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst, distribution=GammaFamily()).fit(
        sample_continuous_data
    )

    result = wrapper.optimize(h=2, underage_cost=3.0, overage_cost=1.0)

    assert np.allclose(result["critical_ratio"], 0.75)
    assert np.issubdtype(result["y_optimal"].dtype, np.floating)


def test_fit_and_predict_with_x_df(sample_train_data):
    """Tests predicting by passing X_df with actual exogenous features when required, or None if pure lags."""
    base_model = LinearRegression()
    fcst = MLForecast(models=[base_model], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst)
    wrapper.fit(sample_train_data)

    future_df = fcst.make_future_dataframe(h=2)
    future_df["exog_feat"] = 1.0

    pred_df = wrapper.predict(h=2, X_df=future_df, quantiles=[0.5])
    assert isinstance(pred_df, pd.DataFrame)
    assert len(pred_df) == 4


def test_predict_floors_negative_lambdas(sample_train_data, monkeypatch):
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst).fit(sample_train_data)
    raw_predictions = fcst.make_future_dataframe(h=1)
    raw_predictions[wrapper.model_name] = -1.0
    monkeypatch.setattr(fcst, "predict", lambda **kwargs: raw_predictions.copy())

    pred_df = wrapper.predict(h=1, quantiles=[])

    assert np.all(pred_df["lambda_t"] == 1e-6)


def test_optimize_with_costs(sample_train_data_with_costs):
    """Tests inventory optimization workflow using a future DataFrame containing costs and exog features."""
    base_model = LinearRegression()
    fcst = MLForecast(models=[base_model], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst)
    wrapper.fit(sample_train_data_with_costs, static_features=["co", "cu"])

    future_df = fcst.make_future_dataframe(h=2)
    future_df["cu"] = 10.0
    future_df["co"] = 2.0
    future_df["exog_feat"] = 1.0

    opt_df = wrapper.optimize(
        h=2, underage_cost="cu", overage_cost="co", X_df=future_df
    )
    assert isinstance(opt_df, pd.DataFrame)
    assert "critical_ratio" in opt_df.columns
    assert "y_optimal" in opt_df.columns


def test_optimize_with_scalar_costs_without_x_df(sample_train_data):
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst).fit(sample_train_data)

    opt_df = wrapper.optimize(h=2, underage_cost=10.0, overage_cost=2.0)

    assert len(opt_df) == 4
    assert np.allclose(opt_df["critical_ratio"], 10.0 / 12.0)


def test_optimize_accepts_x_df_with_only_cost_columns(sample_train_data):
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst).fit(sample_train_data)
    costs = pd.DataFrame({"cu": [10.0] * 4, "co": [2.0] * 4})

    result = wrapper.optimize(h=2, underage_cost="cu", overage_cost="co", X_df=costs)

    assert len(result) == len(costs)
    assert np.allclose(result["critical_ratio"], 10.0 / 12.0)


def test_optimize_with_cost_dicts(sample_train_data):
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst).fit(sample_train_data)

    opt_df = wrapper.optimize(
        h=2,
        underage_cost={"A": 10.0, "B": 6.0},
        overage_cost={"A": 2.0, "B": 4.0},
    )

    expected = opt_df["unique_id"].map({"A": 10.0 / 12.0, "B": 6.0 / 10.0})
    assert np.allclose(opt_df["critical_ratio"], expected)


def test_pmf_and_marginal_benefit(sample_train_data):
    """Tests PMF and marginal benefit calculations with proper exog features in X_df."""
    base_model = LinearRegression()
    fcst = MLForecast(models=[base_model], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst)
    wrapper.fit(sample_train_data)

    max_k = 4
    future_df = fcst.make_future_dataframe(h=2)
    future_df["exog_feat"] = 1.0

    pmf_df = wrapper.pmf(h=2, max_k=max_k, X_df=future_df)
    assert isinstance(pmf_df, pd.DataFrame)
    assert f"P(Y={max_k})" in pmf_df.columns

    future_df["cu"] = 10.0
    future_df["co"] = 2.0
    mb_df = wrapper.marginal_benefit(
        h=2, underage_cost="cu", overage_cost="co", max_k=max_k, X_df=future_df
    )
    assert isinstance(mb_df, pd.DataFrame)
    assert f"MB(k={max_k})" in mb_df.columns
    pmf_values = pmf_df[[f"P(Y={k})" for k in range(max_k + 1)]].to_numpy()
    probability_below_k = np.hstack(
        [np.zeros((len(pmf_df), 1)), np.cumsum(pmf_values, axis=1)[:, :-1]]
    )
    expected_marginal_benefit = 10.0 * (1.0 - probability_below_k) - (
        2.0 * probability_below_k
    )
    actual_marginal_benefit = mb_df[[f"MB(k={k})" for k in range(max_k + 1)]].to_numpy()
    assert np.allclose(actual_marginal_benefit, expected_marginal_benefit)


def test_pmf_rejects_negative_max_k(sample_train_data):
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst).fit(sample_train_data)
    with pytest.raises(ValueError, match="non-negative integer"):
        wrapper.pmf(h=1, max_k=-1)


def test_first_stage_evaluator_metrics():
    df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "a"],
            "ds": pd.date_range("2024-01-01", periods=3),
            "y": [0.0, 2.0, 4.0],
            "lambda_t": [1.0, 1.0, 5.0],
        }
    )
    result = FirstStageForecasterEvaluator.evaluate(df)

    assert result.loc["WAPE", "Metrics"] == pytest.approx(50.0)
    assert result.loc["PBias", "Metrics"] == pytest.approx(16.67)
    assert result.loc["Score", "Metrics"] == pytest.approx(66.6667)
    assert result.loc["Forecast Instability", "Metrics"] == pytest.approx(200.0)
    assert result.loc["False Demand on Zero-Days (Avg Pred)", "Metrics"] == 1.0
    assert result.loc["Peak Demand Deviation (%)", "Metrics"] == 0.0


def test_first_stage_forecast_instability_does_not_cross_series():
    df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": [1, 2, 1, 2],
            "y": [1.0, 1.0, 10.0, 10.0],
            "lambda_t": [1.0, 1.0, 10.0, 10.0],
        }
    )
    result = FirstStageForecasterEvaluator.evaluate(df)

    assert result.loc["Forecast Instability", "Metrics"] == pytest.approx(0.0)


def test_first_stage_calibration_table():
    df = pd.DataFrame(
        {
            "h": [1, 1, 2, 2],
            "y": [0.0, 2.0, 4.0, 6.0],
            "lambda_t": [0.5, 1.5, 4.5, 5.5],
        }
    )
    calibration = FirstStageForecasterEvaluator.calibration_table(df, n_bins=2)
    assert calibration["Count"].sum() == len(df)
    assert len(calibration) == 2
    assert calibration.iloc[0]["Mean_Residual"] == pytest.approx(0.0)


def test_first_stage_evaluator_rejects_non_positive_mean():
    df = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": [1, 2],
            "y": [0.0, 1.0],
            "lambda_t": [0.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="strictly positive"):
        FirstStageForecasterEvaluator.evaluate(df)


def test_tsf_evaluator_ignores_nan_pairs_for_loss_and_coverage():
    df = pd.DataFrame({"y": [1.0, 2.0, np.nan], "q_50": [1.0, 3.0, 0.0]})
    result = TwoStageForecasterEvaluator.evaluate(df, quantiles=[0.5])

    assert result.loc["q_50", "Pinball Loss"] == pytest.approx(0.25)
    assert result.loc["q_50", "Empirical Coverage"] == 1.0


def test_tsf_evaluator_rejects_invalid_quantile():
    df = pd.DataFrame({"y": [1.0], "q_100": [1.0]})
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        TwoStageForecasterEvaluator.evaluate(df, quantiles=[1.0])


@pytest.mark.parametrize("family", [NegativeBinomialFamily(), GammaFamily()])
def test_family_fit_dispersion_rejects_misaligned_or_non_finite_means(family):
    target = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="same shape"):
        family.fit_dispersion(target, np.array([1.0]))
    with pytest.raises(ValueError, match="must be finite"):
        family.fit_dispersion(target, np.array([1.0, np.nan]))


@pytest.mark.parametrize(
    ("family", "target"),
    [
        (NegativeBinomialFamily(min_size=0.1, max_size=8.0), [0, 1, 3, 2]),
        (GammaFamily(min_shape=0.1, max_shape=20.0), [0.8, 1.5, 2.2, 3.0]),
    ],
)
def test_family_fit_dispersion_uses_configured_bounds(family, target):
    result = family.fit_dispersion(np.asarray(target), np.array([1.0, 1.5, 2.0, 2.5]))

    assert family.dispersion_bounds[0] <= result <= family.dispersion_bounds[1]


def test_family_fit_dispersion_reports_optimizer_failure(monkeypatch):
    class FailedResult:
        success = False
        message = "did not converge"
        x = np.nan
        fun = np.inf

    monkeypatch.setattr(
        tsf_family_module, "minimize_scalar", lambda *args, **kwargs: FailedResult()
    )

    with pytest.raises(RuntimeError, match="Dispersion optimization failed"):
        NegativeBinomialFamily().fit_dispersion(np.array([0, 1]), np.array([0.5, 1.0]))


@pytest.mark.parametrize("family", [NegativeBinomialFamily(), GammaFamily()])
@pytest.mark.parametrize("dispersion", [0.0, -1.0, np.nan, np.inf])
def test_family_negative_log_likelihood_rejects_invalid_dispersion(family, dispersion):
    target = np.array([1.0, 2.0])
    means = np.array([1.0, 2.0])

    assert family.negative_log_likelihood(dispersion, target, means) == 1e10


@pytest.mark.parametrize(
    ("means", "dispersions", "message"),
    [
        ([1.0, 2.0], [1.0], "aligned one-dimensional"),
        ([[1.0, 2.0]], [[1.0, 2.0]], "aligned one-dimensional"),
        ([1.0, np.nan], [1.0, 1.0], "must be finite"),
        ([1.0, 2.0], [1.0, np.inf], "must be finite"),
        ([0.0, 2.0], [1.0, 1.0], "strictly positive"),
        ([1.0, 2.0], [-1.0, 1.0], "strictly positive"),
    ],
)
def test_parametric_distribution_rejects_invalid_parameters(
    means, dispersions, message
):
    with pytest.raises(ValueError, match=message):
        GammaPredictiveDistribution(means, dispersions)


@pytest.fixture
def gamma_distribution():
    return GammaPredictiveDistribution([2.0, 4.0], [3.0, 5.0])


@pytest.fixture
def count_distribution():
    return NegativeBinomialPredictiveDistribution([2.0, 4.0], [3.0, 5.0])


def test_distribution_aligns_scalar_row_grid_and_matrix(gamma_distribution):
    assert gamma_distribution.cdf(2.0).shape == (2,)
    assert gamma_distribution.cdf([2.0, 4.0]).shape == (2,)
    assert gamma_distribution.cdf([1.0, 2.0, 3.0]).shape == (2, 3)
    assert gamma_distribution.cdf([[1.0, 2.0], [3.0, 4.0]]).shape == (2, 2)

    with pytest.raises(ValueError, match="scalar, a grid"):
        gamma_distribution.cdf(np.ones((3, 2)))
    with pytest.raises(ValueError, match="finite values"):
        gamma_distribution.cdf(np.nan)


@pytest.mark.parametrize("quantile", [-0.01, 1.01])
def test_predictive_distributions_reject_invalid_quantiles(
    gamma_distribution, count_distribution, quantile
):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        gamma_distribution.ppf(quantile)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        count_distribution.ppf(quantile)


@pytest.mark.parametrize("coverage", [0.0, 1.0, -0.1, np.nan, np.inf])
def test_distribution_interval_rejects_invalid_coverage(gamma_distribution, coverage):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        gamma_distribution.interval(coverage)


@pytest.mark.parametrize("n_samples", [0, -1, 1.5, True])
def test_distribution_sample_rejects_invalid_count(gamma_distribution, n_samples):
    with pytest.raises(ValueError, match="positive integer"):
        gamma_distribution.sample(n_samples)


def test_distribution_sample_is_reproducible(gamma_distribution):
    first = gamma_distribution.sample(4, random_state=42)
    second = gamma_distribution.sample(4, random_state=42)

    assert first.shape == (2, 4)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("value", [1.5, np.nan, np.inf])
def test_discrete_pmf_rejects_non_integer_or_non_finite_values(
    count_distribution, value
):
    with pytest.raises(ValueError, match="finite integers"):
        count_distribution.pmf(value)


def test_discrete_distribution_probability_contract(count_distribution):
    values = np.arange(0, 6)
    pmf = count_distribution.pmf(values)

    assert np.allclose(
        pmf,
        count_distribution.cdf(values) - count_distribution.cdf(values - 1),
    )
    quantiles = np.array([[0.1, 0.5, 0.9], [0.1, 0.5, 0.9]])
    projected = count_distribution.ppf(quantiles)
    assert np.all(count_distribution.cdf(projected) >= quantiles)


def test_optimize_preserves_row_aligned_costs(sample_train_data):
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst).fit(sample_train_data)
    costs = pd.DataFrame({"cu": [1.0, 3.0, 8.0, 0.0], "co": [3.0, 1.0, 2.0, 0.0]})

    result = wrapper.optimize(h=2, underage_cost="cu", overage_cost="co", X_df=costs)

    assert np.allclose(result["critical_ratio"], [0.25, 0.75, 0.8, 0.5])


def test_optimize_rejects_bad_cost_frame(sample_train_data):
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst).fit(sample_train_data)

    with pytest.raises(ValueError, match="one row per forecast row"):
        wrapper.optimize(
            h=2,
            underage_cost="cu",
            overage_cost="co",
            X_df=pd.DataFrame({"cu": [1.0], "co": [1.0]}),
        )
    with pytest.raises(KeyError):
        wrapper.optimize(
            h=2,
            underage_cost="missing",
            overage_cost="co",
            X_df=pd.DataFrame({"co": [1.0] * 4}),
        )


@pytest.mark.parametrize("cost", [-1.0, np.nan, np.inf])
def test_optimize_rejects_invalid_costs(sample_train_data, cost):
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst).fit(sample_train_data)

    with pytest.raises(ValueError, match="finite|non-negative"):
        wrapper.optimize(h=1, underage_cost=cost, overage_cost=1.0)


def test_predict_distribution_uses_fallback_for_unknown_series(
    sample_train_data, monkeypatch
):
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst).fit(sample_train_data)
    raw = fcst.make_future_dataframe(h=1).iloc[[0]].copy()
    raw["unique_id"] = "unseen"
    raw[wrapper.model_name] = 2.0
    monkeypatch.setattr(fcst, "predict", lambda **kwargs: raw.copy())

    frame, _ = wrapper.predict_distribution(h=1)

    assert frame.loc[frame.index[0], "r_dispersion"] == pytest.approx(
        wrapper.dispersion_fallback_
    )


def test_first_stage_evaluator_rejects_missing_or_empty_data():
    with pytest.raises(KeyError, match="Columns not found"):
        FirstStageForecasterEvaluator.evaluate(pd.DataFrame({"y": [1.0]}))
    with pytest.raises(ValueError, match="No valid"):
        FirstStageForecasterEvaluator.evaluate(
            pd.DataFrame(
                {
                    "unique_id": ["a"],
                    "ds": [1],
                    "y": [np.nan],
                    "lambda_t": [np.nan],
                }
            )
        )


def test_first_stage_evaluator_handles_all_zero_target():
    data = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": [1, 2],
            "y": [0.0, 0.0],
            "lambda_t": [1.0, 1.0],
        }
    )

    result = FirstStageForecasterEvaluator.evaluate(data)

    assert np.isnan(result.loc["WAPE", "Metrics"])
    assert np.isnan(result.loc["PBias", "Metrics"])


@pytest.mark.parametrize("n_bins", [1, 1.5])
def test_calibration_table_rejects_invalid_bins(n_bins):
    data = pd.DataFrame({"y": [1.0, 2.0], "lambda_t": [1.0, 2.0]})

    with pytest.raises(ValueError, match="greater than or equal to 2"):
        FirstStageForecasterEvaluator.calibration_table(data, n_bins=n_bins)


def test_calibration_table_handles_constant_predictions():
    data = pd.DataFrame({"y": [1.0, 2.0], "lambda_t": [1.5, 1.5]})

    result = FirstStageForecasterEvaluator.calibration_table(data, n_bins=5)

    assert len(result) == 1
    assert result.loc[0, "Calibration Bin"] == "all"


def test_two_stage_evaluator_requires_target_and_skips_missing_quantiles():
    with pytest.raises(KeyError, match="Target column"):
        TwoStageForecasterEvaluator.evaluate(pd.DataFrame({"q_50": [1.0]}))

    result = TwoStageForecasterEvaluator.evaluate(
        pd.DataFrame({"y": [1.0]}), quantiles=(0.5, 0.95)
    )
    assert result.empty


def test_wrapper_joblib_round_trip(sample_train_data, tmp_path):
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst).fit(sample_train_data)
    expected = wrapper.predict(h=2, quantiles=(0.5, 0.95))
    path = tmp_path / "tsf.joblib"

    joblib.dump(wrapper, path)
    restored = joblib.load(path)
    actual = restored.predict(h=2, quantiles=(0.5, 0.95))

    pd.testing.assert_frame_equal(actual, expected)
    assert isinstance(restored.distribution_family_, NegativeBinomialFamily)
    assert restored.dispersion_dict_ == wrapper.dispersion_dict_
    assert restored.dispersion_fallback_ == wrapper.dispersion_fallback_
