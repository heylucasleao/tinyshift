# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import List, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import DecomposeResult


def extract_mstl_components(
    result: DecomposeResult,
    periods: Union[int, List[int]],
) -> pd.DataFrame:
    """
    Convert a statsmodels MSTL decomposition result to a DataFrame.

    Parameters
    ----------
    result : statsmodels.tsa.seasonal.DecomposeResult
        Fitted MSTL decomposition result.
    periods : int or list of int
        Seasonal periods corresponding to the components in ``result``.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the observed data, trend, one column per seasonal
        component, and residuals.

    Raises
    ------
    TypeError
        If ``result`` is not a ``DecomposeResult``.
    ValueError
        If the number of supplied periods does not match the number of seasonal
        components.

    Notes
    -----
    Seasonal columns are named ``seasonal_<period>`` so they can be joined to
    the component frames used by DMSTL.
    """
    if not isinstance(result, DecomposeResult):
        raise TypeError(
            "Expected 'result' to be a statsmodels DecomposeResult, got "
            f"{type(result).__name__}."
        )

    period_list = [periods] if isinstance(periods, int) else list(periods)
    components = pd.DataFrame(
        {
            "data": result.observed,
            "trend": result.trend,
        }
    )
    seasonal = np.asarray(result.seasonal)
    n_seasonal_channels = 1 if seasonal.ndim == 1 else seasonal.shape[1]
    if len(period_list) != n_seasonal_channels:
        raise ValueError(
            f"Number of provided periods ({len(period_list)}) does not match "
            "the number of seasonal components in result "
            f"({n_seasonal_channels})."
        )

    if seasonal.ndim == 1:
        components[f"seasonal_{period_list[0]}"] = seasonal
    else:
        for index, period in enumerate(period_list):
            components[f"seasonal_{period}"] = seasonal[:, index]

    components["resid"] = result.resid
    return components


def seasonal_strength(
    seasonal_component: Union[np.ndarray, List[float], pd.Series],
    residuals: Union[np.ndarray, List[float], pd.Series],
) -> float:
    """
    Calculate Hyndman's seasonal-strength measure from decomposition components.

    Parameters
    ----------
    seasonal_component : Union[np.ndarray, List[float], pd.Series]
        Seasonal component extracted from an MSTL decomposition.
    residuals : Union[np.ndarray, List[float], pd.Series]
        Residual component from the same decomposition.

    Returns
    -------
    float
        Seasonal-strength score between 0 and 1.

    Notes
    -----
    The score measures the relative reduction in variance after adding the
    seasonal component to the residuals. It is a post-decomposition diagnostic,
    not a seasonal-period detection test.
    """
    seasonal_component = np.asarray(seasonal_component, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)

    var_resid = np.var(residuals, ddof=1)
    var_seas_resid = np.var(seasonal_component + residuals, ddof=1)
    return float(
        max(0.0, 1.0 - (var_resid / var_seas_resid)) if var_seas_resid > 0 else 0.0
    )
