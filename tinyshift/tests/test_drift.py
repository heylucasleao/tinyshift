import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from tinyshift.drift import (
    CatDrift,
    CategoricalDriftAnalyzer,
    ConDrift,
    ContinuousDriftAnalyzer,
    DriftResult,
)


class TestConDrift:
    def test_permutation_and_structured_result(self):
        detector = ConDrift(n_resamples=99, random_state=7).fit(np.linspace(0, 1, 40))
        result = detector.predict(np.linspace(10, 11, 20))
        assert isinstance(result, DriftResult)
        assert result.score > result.threshold
        assert result.p_value == pytest.approx(0.01)
        assert result.drift is True
        assert (result.reference_size, result.current_size) == (40, 20)

    def test_normalization_is_scale_independent(self):
        small = ConDrift(n_resamples=19).fit([0.0, 1.0, 2.0])
        large = ConDrift(n_resamples=19).fit([0.0, 10.0, 20.0])
        assert small.predict([1.0, 2.0, 3.0]).score == pytest.approx(
            large.predict([10.0, 20.0, 30.0]).score
        )

    @pytest.mark.parametrize("values", [["bad", "data"], [1.0, np.inf]])
    def test_invalid_continuous_values_are_rejected(self, values):
        with pytest.raises(ValueError, match="numeric|finite"):
            ConDrift().fit(values)

    def test_lifecycle_and_minimum_sizes_are_validated(self):
        with pytest.raises(NotFittedError):
            ConDrift().predict([1.0, 2.0])
        with pytest.raises(ValueError, match="at least 2"):
            ConDrift().fit([1.0])
        with pytest.raises(ValueError, match="at least 2"):
            ConDrift().fit([1.0, 2.0]).predict([1.0])

    def test_estimator_clone_preserves_configuration(self):
        detector = ConDrift(random_state=4)
        assert clone(detector).get_params() == detector.get_params()

    @pytest.mark.parametrize(
        ("parameter", "value", "message"),
        [
            ("alpha", 0, "alpha"),
            ("alpha", 1, "alpha"),
            ("alpha", "0.05", "alpha"),
            ("n_resamples", 0, "n_resamples"),
            ("n_resamples", True, "n_resamples"),
            ("min_reference_size", 0, "min_reference_size"),
            ("min_current_size", 1.5, "min_current_size"),
        ],
    )
    def test_invalid_parameters_are_rejected(self, parameter, value, message):
        detector = ConDrift(**{parameter: value})
        with pytest.raises(ValueError, match=message):
            detector.fit([1.0, 2.0])

    def test_seed_makes_permutation_inference_reproducible(self):
        reference = np.linspace(0, 1, 20)
        current = np.linspace(0.25, 1.25, 10)
        first = ConDrift(n_resamples=31, random_state=12).fit(reference).predict(current)
        second = ConDrift(n_resamples=31, random_state=12).fit(reference).predict(current)
        assert first == second

    def test_identical_samples_have_no_drift(self):
        sample = np.arange(10, dtype=float)
        result = ConDrift(n_resamples=19, random_state=2).fit(sample).predict(sample)
        assert result.score == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0)
        assert result.drift is False

    def test_multidimensional_samples_are_rejected(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            ConDrift().fit([[1.0, 2.0], [3.0, 4.0]])


class TestCatDrift:
    def test_jensen_shannon_distance(self):
        detector = CatDrift(n_resamples=19).fit(["a", "a", "b", "b"])
        assert detector.predict(["a", "b", "b", "b"]).score > 0

    def test_unseen_categories_contribute_to_distance(self):
        detector = CatDrift(min_current_size=1, n_resamples=19).fit(["known", "known"])
        assert detector.predict(["new"]).score == pytest.approx(1.0)

    def test_missing_values_are_rejected(self):
        with pytest.raises(ValueError, match="missing"):
            CatDrift().fit(["a", None])

    def test_permutation_supports_string_categories(self):
        detector = CatDrift(n_resamples=20, random_state=1).fit(
            ["a", "a", "b", "b", "c"]
        )
        result = detector.predict(["a", "b"])
        assert np.isfinite(result.threshold)
        assert 0 < result.p_value <= 1
        assert isinstance(result.drift, bool)

    def test_identical_samples_have_zero_distance_and_no_drift(self):
        sample = ["a", "a", "b", "c"]
        result = CatDrift(n_resamples=19, random_state=2).fit(sample).predict(sample)
        assert result.score == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0)
        assert result.drift is False

    def test_unhashable_categories_are_rejected(self):
        values = np.empty(2, dtype=object)
        values[:] = [{"category": "a"}, {"category": "b"}]
        with pytest.raises(ValueError, match="hashable"):
            CatDrift().fit(values)


def _panel(a_values, b_values):
    return pd.DataFrame(
        {
            "entity": ["A"] * len(a_values) + ["B"] * len(b_values),
            "value": list(a_values) + list(b_values),
        }
    )


class TestDriftAnalyzers:
    def test_analyzer_rejects_wrong_detector_type(self):
        with pytest.raises(TypeError, match="ConDrift"):
            ContinuousDriftAnalyzer(CatDrift())

    def test_summary_requires_prediction(self):
        analyzer = ContinuousDriftAnalyzer().fit(
            _panel([1.0, 2.0], [3.0, 4.0]), "entity", "value"
        )
        with pytest.raises(NotFittedError):
            analyzer.summary()

    @pytest.mark.parametrize(
        "frame, error, message",
        [
            ([], TypeError, "DataFrame"),
            (pd.DataFrame(), ValueError, "missing required columns"),
            (
                pd.DataFrame({"entity": [None, None], "value": [1.0, 2.0]}),
                ValueError,
                "must not be missing",
            ),
        ],
    )
    def test_reference_frame_is_validated(self, frame, error, message):
        with pytest.raises(error, match=message):
            ContinuousDriftAnalyzer().fit(frame, "entity", "value")

    def test_continuous_analyzer_fits_independent_detectors(self):
        reference = _panel(np.linspace(0, 1, 40), np.linspace(100, 101, 40))
        current = _panel(np.linspace(10, 11, 20), np.linspace(100, 101, 20))
        analyzer = ContinuousDriftAnalyzer(
            ConDrift(n_resamples=99, random_state=7)
        ).fit(reference, id_col="entity", target_col="value")
        result = analyzer.predict(current)
        assert list(result.columns) == [
            "entity",
            "score",
            "threshold",
            "p_value",
            "drift",
            "reference_size",
            "current_size",
        ]
        assert list(result["entity"]) == ["A", "B"]
        assert bool(result.loc[result["entity"] == "A", "drift"].iloc[0])
        assert not bool(result.loc[result["entity"] == "B", "drift"].iloc[0])
        pd.testing.assert_frame_equal(analyzer.summary(), result)
        assert analyzer.detectors_["A"] is not analyzer.detectors_["B"]

    def test_categorical_analyzer_uses_one_result_per_id(self):
        analyzer = CategoricalDriftAnalyzer(
            CatDrift(n_resamples=99, random_state=7)
        ).fit(_panel(["x"] * 40, ["z"] * 40), "entity", "value")
        result = analyzer.predict(_panel(["y"] * 20, ["z"] * 20))
        assert len(result) == 2
        assert result.set_index("entity").loc["A", "drift"]
        assert not result.set_index("entity").loc["B", "drift"]

    def test_unknown_current_id_is_rejected(self):
        analyzer = ContinuousDriftAnalyzer(ConDrift()).fit(
            _panel([1.0, 2.0], [3.0, 4.0]), "entity", "value"
        )
        unknown = pd.DataFrame({"entity": ["C", "C"], "value": [1.0, 2.0]})
        with pytest.raises(ValueError, match="No reference distribution"):
            analyzer.predict(unknown)

    def test_missing_reference_ids_do_not_appear_in_current_result(self):
        analyzer = ContinuousDriftAnalyzer(ConDrift()).fit(
            _panel([1.0, 2.0], [3.0, 4.0]), "entity", "value"
        )
        current = pd.DataFrame({"entity": ["A", "A"], "value": [1.0, 2.0]})
        assert analyzer.predict(current)["entity"].tolist() == ["A"]
