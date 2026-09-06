def test_focused_packages_expose_their_public_apis():
    from tinyshift.forecasting import (
        DTLWrapper,
        LogNormalFamily,
        TwoStageForecasterWrapper,
        WeibullFamily,
        hfi,
        wape,
    )
    from tinyshift.preprocessing import RobustGaussianScaler

    assert DTLWrapper.__module__.startswith("tinyshift.forecasting")
    assert TwoStageForecasterWrapper.__module__.startswith("tinyshift.forecasting")
    assert LogNormalFamily.__module__.startswith("tinyshift.forecasting")
    assert WeibullFamily.__module__.startswith("tinyshift.forecasting")
    assert hfi.__module__ == "tinyshift.forecasting.stabilization"
    assert wape.__module__ == "tinyshift.forecasting.metrics"
    assert RobustGaussianScaler.__module__.startswith("tinyshift.preprocessing")


def test_series_public_api_does_not_leak_module_dependencies():
    from tinyshift import series

    assert not hasattr(series, "np")
    assert not hasattr(series, "pd")
    assert not hasattr(series, "List")
    assert not hasattr(series, "hfi")
    assert not hasattr(series, "wape")
