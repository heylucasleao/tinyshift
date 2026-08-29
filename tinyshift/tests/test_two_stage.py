import numpy as np
import pandas as pd
import pytest
from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression

import tinyshift.modelling.two_stage.wrapper as two_stage_wrapper_module
from tinyshift.modelling import (
    FirstStageForecasterEvaluator,
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
        two_stage_wrapper_module,
        "minimize_scalar",
        lambda *args, **kwargs: FailedResult(),
    )
    wrapper = TwoStageForecasterWrapper(fcst=None)

    with pytest.raises(RuntimeError, match="Dispersion optimization failed"):
        wrapper._estimate_r(np.array([1.0]), np.array([1.0]))


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
        wrapper._compute_critical_quantile(
            cu=np.array([-1.0]), co=np.array([2.0])
        )


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
    actual_marginal_benefit = mb_df[
        [f"MB(k={k})" for k in range(max_k + 1)]
    ].to_numpy()
    assert np.allclose(actual_marginal_benefit, expected_marginal_benefit)


def test_pmf_rejects_negative_max_k(sample_train_data):
    fcst = MLForecast(models=[LinearRegression()], freq="D", lags=[1])
    wrapper = TwoStageForecasterWrapper(fcst=fcst).fit(sample_train_data)
    with pytest.raises(ValueError, match="non-negative integer"):
        wrapper.pmf(h=1, max_k=-1)


def test_first_stage_evaluator_metrics():
    df = pd.DataFrame({"y": [0.0, 2.0, 4.0], "lambda_t": [1.0, 1.0, 5.0]})
    result = FirstStageForecasterEvaluator.evaluate(df)

    assert result.loc["PBias", "Metrics"] == pytest.approx(16.67)
    assert result.loc["False Demand on Zero-Days (Avg Pred)", "Metrics"] == 1.0
    assert result.loc["Peak Demand Deviation (%)", "Metrics"] == 0.0


def test_two_stage_evaluator_ignores_nan_pairs_for_loss_and_coverage():
    df = pd.DataFrame({"y": [1.0, 2.0, np.nan], "q_50": [1.0, 3.0, 0.0]})
    result = TwoStageForecasterEvaluator.evaluate(df, quantiles=[0.5])

    assert result.loc["q_50", "Pinball Loss"] == pytest.approx(0.25)
    assert result.loc["q_50", "Empirical Coverage"] == 1.0


def test_two_stage_evaluator_rejects_invalid_quantile():
    df = pd.DataFrame({"y": [1.0], "q_100": [1.0]})
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        TwoStageForecasterEvaluator.evaluate(df, quantiles=[1.0])
