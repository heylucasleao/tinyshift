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


def test_series_metrics_live_in_their_domain_modules():
    from tinyshift.series import (
        foreca,
        PAMIAnalyzer,
        create_pami_lags,
        permutation_auto_mutual_information,
        permutation_entropy,
        theoretical_limit,
    )

    assert foreca.__module__ == "tinyshift.series.spectral"
    assert permutation_entropy.__module__ == "tinyshift.series.entropy"
    assert theoretical_limit.__module__ == "tinyshift.series.entropy"
    assert (
        permutation_auto_mutual_information.__module__
        == "tinyshift.series.dependence"
    )
    assert PAMIAnalyzer.__module__ == "tinyshift.series.analyzers.pami"
    assert create_pami_lags.__module__ == "tinyshift.series.analyzers.pami"
