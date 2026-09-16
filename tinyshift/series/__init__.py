"""Time-series analysis, diagnostics, and profiling tools."""

from .analyzers import (
    IntermittencyAnalyzer,
    PAMIAnalyzer,
    PAMIResult,
    RegularityAnalyzer,
    SeasonalityAnalyzer,
    TemporalChange,
    TemporalRegime,
    TemporalStabilityAnalyzer,
    TemporalStabilityResult,
    TrendAnalyzer,
    VarianceRatioAnalyzer,
    create_pami_lags,
)
from .dependence import permutation_auto_mutual_information
from .diagnostic import (
    harmonic_significance,
    trend_significance,
    variance_ratio,
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
    "PAMIAnalyzer",
    "PAMIResult",
    "RegularityAnalyzer",
    "SeasonalityAnalyzer",
    "TemporalChange",
    "TemporalRegime",
    "TemporalStabilityAnalyzer",
    "TemporalStabilityResult",
    "TrendAnalyzer",
    "VarianceRatioAnalyzer",
    "create_pami_lags",
    "foreca",
    "harmonic_significance",
    "permutation_auto_mutual_information",
    "permutation_entropy",
    "regularity_index",
    "sample_entropy",
    "spectral_concentration",
    "theoretical_limit",
    "trend_significance",
    "variance_ratio",
]
