# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import List, Union

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import DecomposeResult


def detrend(
    df: pd.DataFrame,
    frac: float = 0.2,
    robust: bool = True,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pd.DataFrame:
    """
    Decompose a Nixtla-format panel into LOWESS trend and residual.

    Parameters
    ----------
    df : pd.DataFrame
        Nixtla long-format panel with identifier, timestamp and target columns.
        Additional columns are preserved. Missing values in the target are linearly
        interpolated only while estimating the trend; original values are
        preserved in the output.
    frac : float, default=0.2
        Fraction of observations used for each local LOWESS regression.
        Larger values produce a smoother trend.
    robust : bool, default=True
        Whether to perform robustifying LOWESS iterations that down-weight
        outliers.
    id_col : str, default="unique_id"
        Column identifying each time series.
    time_col : str, default="ds"
        Column containing timestamps or integer time steps.
    target_col : str, default="y"
        Column containing the observed target values.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``trend`` and ``detrended`` appended. Rows retain
        their original order.
    """
    if not 0 < frac <= 1:
        raise ValueError("frac must be greater than 0 and less than or equal to 1.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame in Nixtla long format.")

    required = [id_col, time_col, target_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}.")

    result = df.copy()
    counts = result.groupby(id_col, observed=True)[target_col].transform("count")
    if (counts < 2).any():
        raise ValueError(
            "Each unique_id series must contain at least two observations."
        )

    result = result.sort_values([id_col, time_col])
    clean_series = result.groupby(id_col, observed=True)[target_col].transform(
        lambda group: group.interpolate(method="linear", limit_direction="both")
    )
    iterations = 3 if robust else 0

    def _apply_lowess(series: pd.Series) -> pd.Series:
        values = series.to_numpy(dtype=float)
        time_index = np.arange(len(values))
        trend = lowess(
            values,
            time_index,
            frac=frac,
            it=iterations,
            return_sorted=False,
        )
        return pd.Series(trend, index=series.index)

    result["trend"] = clean_series.groupby(result[id_col], observed=True).transform(
        _apply_lowess
    )
    result["detrended"] = result[target_col] - result["trend"]

    return result.loc[df.index]


def extract_mstl_components(
    result: DecomposeResult,
    periods: Union[int, List[int]],
) -> pd.DataFrame:
    """Convert a statsmodels MSTL decomposition result to a DataFrame."""
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
