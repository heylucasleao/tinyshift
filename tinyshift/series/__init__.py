"""Time-series analysis, diagnostics, decomposition, and profiling tools."""

from .decomposition import detrend, extract_mstl_components
from .diagnostic import hurst_exponent, seasonal_significance, trend_significance
from .forecastability import (
    foreca,
    permutation_auto_mutual_information,
    permutation_entropy,
    regularity_index,
    sample_entropy,
    select_pami_lag,
    spectral_concentration,
    theoretical_limit,
)
from .intermittency import IntermittencyAnalyzer
from .outlier import bollinger_bands, hampel_filter
from .profiler import SeriesProfiler
from .seasonality import SeasonalPeriodDetector

__all__ = [
    "IntermittencyAnalyzer",
    "SeasonalPeriodDetector",
    "SeriesProfiler",
    "bollinger_bands",
    "detrend",
    "extract_mstl_components",
    "foreca",
    "hampel_filter",
    "hurst_exponent",
    "permutation_auto_mutual_information",
    "permutation_entropy",
    "regularity_index",
    "sample_entropy",
    "seasonal_significance",
    "select_pami_lag",
    "spectral_concentration",
    "theoretical_limit",
    "trend_significance",
]
