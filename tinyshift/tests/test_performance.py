"""Behavioral tests for direct loss estimation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from tinyshift.performance import (
    ConfidenceBasedPerformanceAnalyzer,
    ConfidenceBasedPerformanceEstimator,
    DirectLossAnalyzer,
    DirectLossEstimator,
)


def test_direct_loss_estimator_aggregates_rmse_after_mean_squared_loss():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y_pred = np.zeros(8)
    y_true = np.full(8, 2.0)
    estimator = DirectLossEstimator(
        metric="rmse", estimator=DummyRegressor(strategy="mean")
    ).fit(X, y_true, y_pred)

    assert np.all(estimator.estimate_loss(X[:3], y_pred[:3]) == 4.0)
    assert estimator.estimate(X[:3], y_pred[:3]) == 2.0
    assert estimator.aggregate([1.0, 9.0]) == pytest.approx(np.sqrt(5.0))


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
    assert result["reference_realized"].tolist() == [3.0, 1.0]
    assert result["current_estimated"].tolist() == [3.0, 1.0]
    assert result["reference_size"].tolist() == [2, 2]
    assert not result["degradation"].any()
    result.loc[0, "current_estimated"] = -10
    assert analyzer.summary().loc[0, "current_estimated"] == 3.0


def test_analyzer_requires_fitted_ids_and_current_predictions():
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


def test_analyzer_flags_higher_estimated_loss_without_current_targets():
    reference = pd.DataFrame(
        {"unique_id": ["A"] * 8, "x": np.arange(8.0), "y": np.arange(8.0), "y_pred": np.zeros(8)}
    )
    current = pd.DataFrame({"unique_id": ["A"], "x": [10.0], "y_pred": [0.0]})
    analyzer = DirectLossAnalyzer(
        DirectLossEstimator(metric="mae", estimator=LinearRegression())
    ).fit(reference, feature_cols=["x"])

    row = analyzer.predict(current).iloc[0]
    assert row["reference_estimated"] == pytest.approx(6.5)
    assert row["current_estimated"] == pytest.approx(10.0)
    assert row["estimated_delta"] == pytest.approx(3.5)
    assert bool(row["degradation"])


def test_confidence_estimator_binary_expected_confusion_and_f1():
    probabilities = np.array([[0.1, 0.9], [0.8, 0.2]])
    fitted = ConfidenceBasedPerformanceEstimator(metric="f1").fit(
        np.array([1, 0]), probabilities, classes=[0, 1]
    )
    confusion = fitted.expected_confusion_matrix(probabilities)

    np.testing.assert_allclose(confusion, [[0.8, 0.2], [0.1, 0.9]])
    assert fitted.reference_realized_ == 1.0
    assert fitted.estimate(probabilities) == pytest.approx(1.8 / 2.1)


def test_confidence_estimator_multiclass_macro_and_probability_validation():
    probabilities = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    )
    fitted = ConfidenceBasedPerformanceEstimator(metric="accuracy").fit(
        np.array(["a", "b", "c"]), probabilities, classes=["a", "b", "c"]
    )
    assert fitted.estimate(probabilities) == pytest.approx(0.8)
    macro = ConfidenceBasedPerformanceEstimator(metric="f1").fit(
        np.array(["a", "b", "c"]), probabilities, classes=["a", "b", "c"]
    )
    assert macro.estimate(probabilities) == pytest.approx(0.8)
    with pytest.raises(ValueError, match="sum to one"):
        fitted.estimate([[0.3, 0.3, 0.3]])
    with pytest.raises(ValueError, match="Expected 3"):
        fitted.estimate([[0.5, 0.5]])


def test_confidence_analyzer_estimates_without_current_labels():
    reference = pd.DataFrame(
        {
            "unique_id": ["A", "A", "B", "B"],
            "y": [0, 1, 0, 1],
            "p0": [0.8, 0.2, 0.8, 0.2],
            "p1": [0.2, 0.8, 0.2, 0.8],
        }
    )
    current = pd.DataFrame(
        {"unique_id": ["B", "A"], "p0": [0.55, 0.95], "p1": [0.45, 0.05]}
    )
    analyzer = ConfidenceBasedPerformanceAnalyzer().fit(
        reference, probability_cols={0: "p0", 1: "p1"}
    )
    result = analyzer.predict(current)

    assert result["unique_id"].tolist() == ["B", "A"]
    assert result["reference_estimated"].tolist() == pytest.approx([0.8, 0.8])
    assert result["current_estimated"].tolist() == pytest.approx([0.55, 0.95])
    assert result["degradation"].tolist() == [True, False]
    with pytest.raises(ValueError, match="No reference performance"):
        analyzer.predict(pd.DataFrame({"unique_id": ["C"], "p0": [0.5], "p1": [0.5]}))
