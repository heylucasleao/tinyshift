# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Union

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess


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
    df : pandas.DataFrame
        Panel containing one row per observation and the identifier, time, and
        target columns.
    frac : float, default=0.2
        Fraction of observations used when estimating each LOWESS neighborhood.
    robust : bool, default=True
        Whether to perform robust LOWESS iterations.
    id_col : str, default="unique_id"
        Name of the series identifier column.
    time_col : str, default="ds"
        Name of the time column.
    target_col : str, default="y"
        Name of the target column.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with ``trend`` and ``detrended`` columns added.

    Raises
    ------
    TypeError
        If ``df`` is not a pandas DataFrame.
    ValueError
        If ``frac`` is outside (0, 1], required columns are missing, or a
        series contains fewer than two observations.

    Notes
    -----
    Missing target values are linearly interpolated for trend estimation. The
    returned rows preserve the original order and index of ``df``.
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
