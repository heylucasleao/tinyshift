"""Time-series analysis, diagnostics, and profiling tools."""

from .dependence import permutation_auto_mutual_information
from .diagnostic import (
    harmonic_significance,
    trend_significance,
    variance_ratio,
)
from .analyzers import (
    IntermittencyAnalyzer,
    PAMIAnalyzer,
    PAMIResult,
    RegularityAnalyzer,
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
from .spectral import foreca, spectral_concentration

__all__ = [
    "IntermittencyAnalyzer",
    "SeasonalityAnalyzer",
    "foreca",
    "permutation_auto_mutual_information",
    "permutation_entropy",
    "regularity_index",
    "sample_entropy",
    "harmonic_significance",
    "PAMIAnalyzer",
    "PAMIResult",
    "RegularityAnalyzer",
    "TrendAnalyzer",
    "create_pami_lags",
    "spectral_concentration",
    "theoretical_limit",
    "trend_significance",
    "variance_ratio",
    "VarianceRatioAnalyzer",
]
