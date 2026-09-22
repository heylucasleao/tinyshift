"""Behavioral tests for squared direct loss estimation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError

from tinyshift.performance import DirectLossAnalyzer, DirectLossEstimator


def test_direct_loss_estimator_models_mse_and_binary_brier():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y_pred = np.full(8, 0.25)
    y_true = np.ones(8)
    estimator = DirectLossEstimator(estimator=DummyRegressor(strategy="mean"))
    estimator.fit(X, y_true, y_pred)

    np.testing.assert_allclose(estimator.observed_loss(y_true, y_pred), 0.5625)
    np.testing.assert_allclose(estimator.estimate_loss(X[:3], y_pred[:3]), 0.5625)
    assert estimator.estimate(X[:3], y_pred[:3]) == pytest.approx(0.5625)
    assert estimator.aggregate([1.0, 9.0]) == 5.0


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
        {"unique_id": ["B", "A"], "x": [20.0, 21.0], "y_pred": [0.0, 0.0]}
    )
    analyzer = DirectLossAnalyzer(
        DirectLossEstimator(estimator=DummyRegressor(strategy="mean"))
    ).fit(reference, feature_cols=["x"])

    result = analyzer.predict(current)
    assert result["unique_id"].tolist() == ["B", "A"]
    assert result["metric"].tolist() == ["mse", "mse"]
    assert result["reference_realized"].tolist() == [9.0, 1.0]
    assert result["current_estimated"].tolist() == [9.0, 1.0]
    assert result["reference_size"].tolist() == [2, 2]
    assert not result["degradation"].any()
    result.loc[0, "current_estimated"] = -10
    assert analyzer.summary().loc[0, "current_estimated"] == 9.0


def test_analyzer_requires_fitted_ids_and_current_predictions():
    with pytest.raises(TypeError, match="DirectLossEstimator"):
        DirectLossAnalyzer(estimator=DummyRegressor())

    analyzer = DirectLossAnalyzer()
    with pytest.raises(NotFittedError):
        analyzer.predict(pd.DataFrame({"unique_id": ["A"]}))

    reference = pd.DataFrame(
        {"unique_id": ["A"] * 8, "x": np.arange(8), "y": np.ones(8), "y_pred": np.zeros(8)}
    )
    analyzer.fit(reference, feature_cols=["x"])
    with pytest.raises(ValueError, match="No reference performance"):
        analyzer.predict(pd.DataFrame({"unique_id": ["B"], "x": [1], "y_pred": [0]}))
    with pytest.raises(ValueError, match="missing required columns"):
        analyzer.predict(pd.DataFrame({"unique_id": ["A"], "x": [1]}))

    renamed = pd.DataFrame({"series": ["A"], "x": [1], "probability": [0.0]})
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
    current = pd.DataFrame({"unique_id": ["A"], "x": [10.0], "y_pred": [0.5]})
    analyzer = DirectLossAnalyzer(
        DirectLossEstimator(estimator=DummyRegressor(strategy="mean"))
    ).fit(reference, feature_cols=["x"])

    row = analyzer.predict(current).iloc[0]
    assert row["reference_realized"] == pytest.approx(0.04)
    assert row["current_estimated"] == pytest.approx(0.04)
    assert row["metric"] == "mse"
    assert not bool(row["degradation"])
