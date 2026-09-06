"""Time-series analysis, diagnostics, decomposition, and profiling tools."""

from .decomposition import detrend, extract_mstl_components
from .dependence import permutation_auto_mutual_information
from .diagnostic import (
    harmonic_significance,
    seasonal_strength,
    trend_significance,
    variance_ratio,
)
from .analyzers import (
    IntermittencyAnalyzer,
    PAMIAnalyzer,
    PAMIResult,
    PredictabilityAnalyzer,
    SeasonalityAnalyzer,
    TrendAnalyzer,
    VarianceRatioAnalyzer,
    create_pami_lags,
)
from .entropy import (
    permutation_entropy,
    regularity_index,
    sample_entropy,
    theoretical_limit,
)
from .outlier import bollinger_bands, hampel_filter
from .spectral import foreca, spectral_concentration

__all__ = [
    "IntermittencyAnalyzer",
    "SeasonalityAnalyzer",
    "bollinger_bands",
    "detrend",
    "extract_mstl_components",
    "foreca",
    "hampel_filter",
    "permutation_auto_mutual_information",
    "permutation_entropy",
    "regularity_index",
    "sample_entropy",
    "harmonic_significance",
    "seasonal_strength",
    "PAMIAnalyzer",
    "PAMIResult",
    "PredictabilityAnalyzer",
    "TrendAnalyzer",
    "create_pami_lags",
    "spectral_concentration",
    "theoretical_limit",
    "trend_significance",
    "variance_ratio",
    "VarianceRatioAnalyzer",
]
