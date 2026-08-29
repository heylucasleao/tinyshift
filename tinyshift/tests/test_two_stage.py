import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from mlforecast import MLForecast
from tinyshift.modelling import TwoStageForecasterWrapper


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


def test_compute_critical_quantile():
    wrapper = TwoStageForecasterWrapper(fcst=None)
    cu = np.array([10.0, 5.0, 0.0])
    co = np.array([2.0, 5.0, 0.0])
    q_star = wrapper._compute_critical_quantile(cu, co)

    assert isinstance(q_star, np.ndarray)
    assert q_star[0] == pytest.approx(10.0 / 12.0)
    assert q_star[1] == pytest.approx(5.0 / 10.0)
    assert q_star[2] == 0.5


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


def test_pmf_and_marginal_cost(sample_train_data):
    """Tests PMF and marginal cost calculations with proper exog features in X_df."""
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
    mc_df = wrapper.marginal_cost(
        h=2, underage_cost="cu", overage_cost="co", max_k=max_k, X_df=future_df
    )
    assert isinstance(mc_df, pd.DataFrame)
    assert f"MC(k={max_k})" in mc_df.columns
