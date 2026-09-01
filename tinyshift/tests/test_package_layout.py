import importlib


def test_focused_packages_expose_their_public_apis():
    from tinyshift.features import relative_strength_index
    from tinyshift.forecasting import DTLWrapper, TwoStageForecasterWrapper
    from tinyshift.preprocessing import RobustGaussianScaler

    assert callable(relative_strength_index)
    assert DTLWrapper.__module__.startswith("tinyshift.forecasting")
    assert TwoStageForecasterWrapper.__module__.startswith("tinyshift.forecasting")
    assert RobustGaussianScaler.__module__.startswith("tinyshift.preprocessing")


def test_modelling_paths_alias_canonical_modules():
    old_family = importlib.import_module("tinyshift.modelling.tsf.family")
    new_family = importlib.import_module(
        "tinyshift.forecasting.probabilistic.family"
    )

    assert old_family is new_family
