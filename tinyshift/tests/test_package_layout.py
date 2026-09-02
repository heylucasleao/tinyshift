def test_focused_packages_expose_their_public_apis():
    from tinyshift.forecasting import DTLWrapper, TwoStageForecasterWrapper
    from tinyshift.preprocessing import RobustGaussianScaler

    assert DTLWrapper.__module__.startswith("tinyshift.forecasting")
    assert TwoStageForecasterWrapper.__module__.startswith("tinyshift.forecasting")
    assert RobustGaussianScaler.__module__.startswith("tinyshift.preprocessing")
