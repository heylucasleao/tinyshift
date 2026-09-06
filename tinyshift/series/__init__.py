"""Time-series analysis, diagnostics, decomposition, and profiling tools."""

from .decomposition import detrend, extract_mstl_components
from .dependence import permutation_auto_mutual_information
from .diagnostic import seasonal_significance, trend_significance, variance_ratio
from .entropy import (
    permutation_entropy,
    regularity_index,
    sample_entropy,
    theoretical_limit,
)
from .intermittency import IntermittencyAnalyzer
from .outlier import bollinger_bands, hampel_filter
from .pami import PAMIAnalyzer, PAMIResult, create_pami_lags
from .profiler import SeriesProfiler
from .seasonality import SeasonalPeriodDetector
from .spectral import foreca, spectral_concentration
from .variance_ratio import VarianceRatioAnalyzer

__all__ = [
    "IntermittencyAnalyzer",
    "SeasonalPeriodDetector",
    "SeriesProfiler",
    "bollinger_bands",
    "detrend",
    "extract_mstl_components",
    "foreca",
    "hampel_filter",
    "permutation_auto_mutual_information",
    "permutation_entropy",
    "regularity_index",
    "sample_entropy",
    "seasonal_significance",
    "PAMIAnalyzer",
    "PAMIResult",
    "create_pami_lags",
    "spectral_concentration",
    "theoretical_limit",
    "trend_significance",
    "variance_ratio",
    "VarianceRatioAnalyzer",
]
