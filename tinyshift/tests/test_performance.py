"""Behavioral tests for squared direct loss estimation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from tinyshift.performance import (
    DirectLossAnalyzer,
    DirectLossEstimator,
    DirectLossResult,
)


def test_direct_loss_estimator_models_mse_and_binary_brier():
    with pytest.raises(TypeError, match="learner"):
        DirectLossEstimator()

    X = np.arange(8, dtype=float).reshape(-1, 1)
    y_pred = np.full(8, 0.25)
    y_true = np.ones(8)
    estimator = DirectLossEstimator(learner=DummyRegressor(strategy="mean"))
    estimator.fit(X, y_true, y_pred)

    np.testing.assert_allclose(estimator.observed_loss(y_true, y_pred), 0.5625)
    np.testing.assert_allclose(estimator.estimate_loss(X[:3], y_pred[:3]), 0.5625)
    assert estimator.estimate(X[:3], y_pred[:3]) == pytest.approx(0.5625)
    assert estimator.aggregate([1.0, 9.0]) == 5.0
    result = estimator.predict(X[:3], y_pred[:3])
    assert isinstance(result, DirectLossResult)
    assert result.reference_realized == pytest.approx(0.5625)
    assert result.reference_estimated == pytest.approx(0.5625)
    assert result.reference_size == 2
    assert result.current_estimated == pytest.approx(0.5625)
    assert result.estimated_delta == pytest.approx(0)
    assert result.relative_delta == pytest.approx(0)
    assert result.degradation_margin == 0.0
    assert result.reference_limit == pytest.approx(0.5625)
    assert not result.degradation
    assert result.current_size == 3


def test_direct_loss_predict_uses_held_out_baseline():
    X = np.arange(8.0).reshape(-1, 1)
    estimator = DirectLossEstimator(
        learner=DummyRegressor(strategy="mean"), fraction=0.25
    ).fit(X, np.array([1.0] * 6 + [3.0] * 2), np.zeros(8))

    result = estimator.predict(X[:2], np.zeros(2))
    assert result.reference_realized == 9.0
    assert result.reference_estimated == 1.0
    assert result.current_estimated == 1.0
    assert result.estimated_delta == 0.0
    assert not result.degradation


def test_direct_loss_predict_requires_fit_and_valid_split():
    estimator = DirectLossEstimator(DummyRegressor())
    with pytest.raises(NotFittedError):
        estimator.predict([[0.0]], [0.0])
    with pytest.raises(ValueError, match="two fitting rows"):
        estimator.fit([[0.0], [1.0]], [0.0, 1.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="two held-out rows"):
        estimator.fit([[0.0], [1.0], [2.0], [3.0]], [0.0] * 4, [0.0] * 4)

    fitted = estimator.fit(np.arange(8.0).reshape(-1, 1), [0.0] * 8, [0.0] * 8)
    with pytest.raises(ValueError, match="two observations"):
        fitted.predict([[0.0]], [0.0])


@pytest.mark.parametrize("method", ["stddev", "mad", "iqr"])
def test_reference_chunk_interval_and_relative_degradation_margin(method):
    X = np.arange(20.0).reshape(-1, 1)
    estimator = DirectLossEstimator(
        LinearRegression(), fraction=0.25, chunk_size=2, interval_method=method
    ).fit(X, np.sqrt(X.ravel()), np.zeros(20))

    small = estimator.predict([[18.0], [18.0]], [0.0, 0.0])
    large = estimator.predict(np.full((1000, 1), 18.0), np.zeros(1000))
    assert len(estimator.reference_chunk_losses_) == 2
    assert small.reference_limit == estimator.reference_limit_
    assert small.reference_estimated == large.reference_estimated
    assert small.estimated_delta > 0
    assert small.relative_delta == pytest.approx(1 / 17)
    assert small.degradation == large.degradation
    above_margin = estimator.predict(
        np.full((1000, 1), 20.0), np.zeros(1000), degradation_margin=0.10
    )
    below_margin = estimator.predict(
        np.full((1000, 1), 20.0), np.zeros(1000), degradation_margin=0.25
    )
    assert above_margin.relative_delta == pytest.approx(3 / 17)
    assert above_margin.degradation_margin == 0.10
    assert above_margin.degradation == (above_margin.current_estimated > above_margin.reference_limit)
    assert not below_margin.degradation
    improved = estimator.predict(np.full((100, 1), 13.0), np.zeros(100))
    assert not improved.degradation
    with pytest.raises(ValueError, match="degradation_margin"):
        estimator.predict([[20.0], [20.0]], [0.0, 0.0], degradation_margin=-0.1)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"chunk_size": 0}, "chunk_size"),
        ({"chunk_size": True}, "chunk_size"),
        ({"interval_method": "invalid"}, "Unsupported method"),
        ({"interval_method": (None, None)}, "finite upper bound"),
    ],
)
def test_direct_loss_rejects_invalid_alert_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DirectLossEstimator(DummyRegressor(), **kwargs).fit(
            np.arange(8.0).reshape(-1, 1), [0.0] * 8, [0.0] * 8
        )


def test_reference_limit_uses_chunk_means_not_individual_losses():
    estimator = DirectLossEstimator(LinearRegression(), fraction=0.5, chunk_size=2)
    X = np.arange(12.0).reshape(-1, 1)
    estimator.fit(X, np.sqrt(X.ravel()), np.zeros(12))
    np.testing.assert_allclose(estimator.reference_chunk_losses_, [6.5, 8.5, 10.5])
    assert estimator.reference_limit_ == pytest.approx(
        np.mean([6.5, 8.5, 10.5]) + 3 * np.std([6.5, 8.5, 10.5])
    )


def test_chunk_size_can_match_operational_batch_size():
    assert DirectLossEstimator(LinearRegression()).chunk_size == 50
    X = np.arange(800.0).reshape(-1, 1)
    estimator = DirectLossEstimator(
        LinearRegression(), fraction=0.5, chunk_size=100
    ).fit(X, np.sqrt(X.ravel()), np.zeros(800))

    assert len(estimator.reference_chunk_losses_) == 4
    expected = estimator.reference_estimated_losses_.reshape(4, 100).mean(axis=1)
    np.testing.assert_allclose(estimator.reference_chunk_losses_, expected)
    result = estimator.predict(np.full((100, 1), 700.0), np.zeros(100))
    assert result.current_size == 100
    assert result.reference_limit == estimator.reference_limit_


def test_analyzer_uses_held_out_reference_and_independent_ids():
    reference = pd.DataFrame(
        {
            "unique_id": ["A"] * 8 + ["B"] * 8,
            "x": np.arange(16, dtype=float),
            "y_pred": np.zeros(16),
            "y": [1.0] * 8 + [3.0] * 8,
        }
    )
    current = pd.DataFrame(
        {
            "unique_id": ["B", "A", "B", "A"],
            "x": [20.0, 21.0, 22.0, 23.0],
            "y_pred": [0.0] * 4,
        }
    )
    analyzer = DirectLossAnalyzer(
        DirectLossEstimator(learner=DummyRegressor(strategy="mean")), fraction=0.25
    ).fit(reference, feature_cols=["x"])

    result = analyzer.predict(current, degradation_margin=0.10)
    assert result["unique_id"].tolist() == ["B", "A"]
    assert "metric" not in result.columns
    assert result["reference_realized"].tolist() == [9.0, 1.0]
    assert result["current_estimated"].tolist() == [9.0, 1.0]
    assert result["reference_size"].tolist() == [2, 2]
    assert result["degradation_margin"].tolist() == [0.10, 0.10]
    assert not result["degradation"].any()
    assert list(analyzer.results_) == ["B", "A"]
    assert isinstance(analyzer.results_["B"], DirectLossResult)
    assert analyzer.results_["B"].current_estimated == 9.0
    result.loc[0, "current_estimated"] = -10
    assert analyzer.summary().loc[0, "current_estimated"] == 9.0

    analyzer.fit(reference, feature_cols=["x"])
    with pytest.raises(NotFittedError):
        analyzer.summary()


def test_analyzer_validates_fraction():
    reference = pd.DataFrame(
        {"unique_id": ["A"] * 4, "x": [0, 1, 2, 3], "y": [0] * 4, "y_pred": [0] * 4}
    )
    with pytest.raises(ValueError, match="fraction"):
        DirectLossAnalyzer(fraction=0).fit(reference, feature_cols=["x"])


def test_analyzer_requires_fitted_ids_and_current_predictions():
    with pytest.raises(TypeError, match="DirectLossEstimator"):
        DirectLossAnalyzer(estimator=DummyRegressor())

    analyzer = DirectLossAnalyzer()
    with pytest.raises(NotFittedError):
        analyzer.predict(pd.DataFrame({"unique_id": ["A"]}))

    reference = pd.DataFrame(
        {
            "unique_id": ["A"] * 8,
            "x": np.arange(8),
            "y": np.ones(8),
            "y_pred": np.zeros(8),
        }
    )
    analyzer.fit(reference, feature_cols=["x"])
    with pytest.raises(ValueError, match="two observations"):
        analyzer.predict(pd.DataFrame({"unique_id": ["A"], "x": [1], "y_pred": [0]}))
    with pytest.raises(ValueError, match="No reference performance"):
        analyzer.predict(pd.DataFrame({"unique_id": ["B"], "x": [1], "y_pred": [0]}))
    with pytest.raises(ValueError, match="missing required columns"):
        analyzer.predict(pd.DataFrame({"unique_id": ["A"], "x": [1]}))

    renamed = pd.DataFrame(
        {"series": ["A", "A"], "x": [1, 2], "probability": [0.0, 0.0]}
    )
    result = analyzer.predict(renamed, id_col="series", prediction_col="probability")
    assert result["series"].tolist() == ["A"]
    assert analyzer.summary().equals(result)


def test_analyzer_estimates_binary_brier_without_current_labels():
    reference = pd.DataFrame(
        {
            "unique_id": ["A"] * 8,
            "x": np.arange(8.0),
            "y": [0, 1] * 4,
            "y_pred": [0.2, 0.8] * 4,
        }
    )
    current = pd.DataFrame(
        {"unique_id": ["A", "A"], "x": [10.0, 11.0], "y_pred": [0.5, 0.5]}
    )
    analyzer = DirectLossAnalyzer(
        DirectLossEstimator(learner=DummyRegressor(strategy="mean"))
    ).fit(reference, feature_cols=["x"])

    row = analyzer.predict(current).iloc[0]
    assert row["reference_realized"] == pytest.approx(0.04)
    assert row["current_estimated"] == pytest.approx(0.04)
    assert "metric" not in row.index
    assert not bool(row["degradation"])
