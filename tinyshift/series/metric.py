# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

import pandas as pd
from typing import Union, List, Literal
import numpy as np
from tinyshift.stats import rolling_window


def wape(
    df: pd.DataFrame,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pd.DataFrame:
    """Calculate Weighted Absolute Percentage Error (WAPE) for multiple models.

    WAPE measures overall forecast accuracy by dividing the sum of absolute
    errors by total actual demand: sum(|y_true - y_pred|) / sum(|y_true|).
    Unlike MAPE, it is resilient to zero-demand periods, making it a standard
    metric in supply chain management.

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation DataFrame containing ground-truth values and model forecasts.
    models : List[str]
        List of column names corresponding to the forecasting models to evaluate.
    id_col : str, default="unique_id"
        Column name identifying unique series or group identifiers.
    target_col : str, default="y"
        Column name containing actual target values.

    Returns
    -------
    pd.DataFrame
        A DataFrame formatted with `id_col`, `metric` label ('wape'), and
        columns for each evaluated model containing their respective WAPE values.

    Notes
    -----
    Interpretation:
    - Measures overall volume deviation relative to total demand.
    - Expressed as a percentage (%): 0.0% represents a perfect forecast.
    - Always non-negative (>= 0%). Lower values indicate higher accuracy.
    - Equivalent to MAE % (MAE divided by mean actual demand).
    - Example: A WAPE of 15.0% means total forecast errors account for 15%
      of total actual volume across the period.
    """
    rows = []

    for uid, group in df.groupby(id_col, observed=True):
        y_true = group[target_col].to_numpy(dtype=np.float64)
        row_dict = {id_col: uid, "metric": "wape"}
        demand = np.sum(np.abs(y_true))

        for model in models:
            y_pred = group[model].to_numpy(dtype=np.float64)
            abs_errors = np.sum(np.abs(y_pred - y_true))

            if demand == 0:
                score = np.nan if abs_errors != 0 else 0.0
            else:
                score = float((abs_errors / demand) * 100)

            row_dict[model] = score

        rows.append(row_dict)

    return pd.DataFrame(rows)


def pbias(
    df: pd.DataFrame,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pd.DataFrame:
    """Calculate Percent Bias (PBias) for multiple models.

    PBias evaluates the systematic directional drift of forecasts by dividing
    total forecast error by total actual demand: sum(y_pred - y_true) / sum(y_true).

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation DataFrame containing ground-truth values and model forecasts.
    models : List[str]
        List of column names corresponding to the forecasting models to evaluate.
    id_col : str, default="unique_id"
        Column name identifying unique series or group identifiers.
    target_col : str, default="y"
        Column name containing actual target values.

    Returns
    -------
    pd.DataFrame
        A DataFrame formatted with `id_col`, `metric` label ('pbias'), and
        columns for each evaluated model containing their respective PBias values.

    Notes
    -----
    Interpretation:
    - Measures systematic overestimation or underestimation tendencies.
    - Expressed as a percentage (%):
      * 0.0%: Perfectly unbiased forecast on aggregate volume.
      * Positive (> 0%): Systematic overestimation (overforecast / excess inventory risk).
      * Negative (< 0%): Systematic underestimation (underforecast / stockout risk).
    - Example: A PBias of +10.0% means the model predicted 10% more volume
      than actual total demand.
    """
    rows = []

    for uid, group in df.groupby(id_col, observed=True):
        y_true = group[target_col].to_numpy(dtype=np.float64)
        row_dict = {id_col: uid, "metric": "pbias"}
        demand = np.sum(y_true)

        for model in models:
            y_pred = group[model].to_numpy(dtype=np.float64)
            bias = np.sum(y_pred - y_true)

            if demand == 0:
                score = np.nan if bias != 0 else 0.0
            else:
                score = float((bias / demand) * 100)

            row_dict[model] = score

        rows.append(row_dict)

    return pd.DataFrame(rows)


def score(
    df: pd.DataFrame,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pd.DataFrame:
    """Calculate combined Performance Score (WAPE + |PBias|) for multiple models.

    Score evaluates overall forecast quality by combining dispersion error (WAPE)
    and absolute directional drift (|PBias|):
    Score = WAPE + |PBias|

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation DataFrame containing ground-truth values and model forecasts.
    models : List[str]
        List of column names corresponding to the forecasting models to evaluate.
    id_col : str, default="unique_id"
        Column name identifying unique series or group identifiers.
    target_col : str, default="y"
        Column name containing actual target values.

    Returns
    -------
    pd.DataFrame
        A DataFrame formatted with `id_col`, `metric` label ('score'), and
        columns for each evaluated model containing their respective Score values.

    Notes
    -----
    Interpretation:
    - Acts as a composite loss metric penalizing both total volume error and systematic bias.
    - Expressed as a percentage (%): 0.0% represents a perfect forecast.
    - Lower values are better.
    - Internally calls `wape` and `pbias` functions to build the composite score.
    """

    df_wape = wape(df=df, models=models, id_col=id_col, target_col=target_col)
    df_pbias = pbias(df=df, models=models, id_col=id_col, target_col=target_col)
    wape_values = df_wape[models].to_numpy()
    pbias_values = df_pbias[models].to_numpy()
    score_values = wape_values + np.abs(pbias_values)
    df_score = df_wape[[id_col]].copy()
    df_score["metric"] = "score"
    df_score[models] = score_values

    return df_score


def rae(
    df: pd.DataFrame,
    models: List[str],
    baseline_col: str,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pd.DataFrame:
    """Calculate Relative Absolute Error (RAE) for multiple models against a baseline.

    RAE measures model efficiency by comparing the Mean Absolute Error (MAE)
    of candidate forecasting models against the MAE of a benchmark baseline
    forecast: sum(|y_true - y_pred|) / sum(|y_true - y_baseline|).

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation DataFrame containing ground-truth values, model forecasts,
        and baseline predictions.
    models : List[str]
        List of column names corresponding to candidate forecasting models to evaluate.
    baseline_col : str
        Column name corresponding to the benchmark baseline forecast
        (e.g., 'naive', 'seasonal_naive', 'moving_average').
    id_col : str, default="unique_id"
        Column name identifying unique series or group identifiers.
    target_col : str, default="y"
        Column name containing actual target values.

    Returns
    -------
    pd.DataFrame
        A DataFrame formatted with `id_col`, `metric` label ('rae'), and
        columns for each evaluated model containing their respective RAE values.

    Notes
    -----
    Interpretation & Forecast Value Added (FVA):
    - Evaluates whether a complex model adds value over a simple baseline.
    - **RAE < 1.0**: Model outperforms baseline (Positive FVA). Lower is better.
      Example: RAE = 0.80 means the model reduced absolute errors by 20% compared to baseline.
    - **RAE = 1.0**: Model performs identically to baseline (No added value).
    - **RAE > 1.0**: Model performs worse than baseline (Negative FVA / destroys value).
      Example: RAE = 1.25 means the model generated 25% more error than a simple baseline.
    """
    rows = []

    for uid, group in df.groupby(id_col, observed=True):
        y_true = group[target_col].to_numpy(dtype=np.float64)
        y_baseline = group[baseline_col].to_numpy(dtype=np.float64)

        mae_baseline = np.sum(np.abs(y_true - y_baseline))
        row_dict = {id_col: uid, "metric": "rae"}

        for model in models:
            y_pred = group[model].to_numpy(dtype=np.float64)
            mae_model = np.sum(np.abs(y_true - y_pred))

            if mae_baseline == 0:
                score = np.nan if mae_model != 0 else 1.0
            else:
                score = float(mae_model / mae_baseline)

            row_dict[model] = score

        rows.append(row_dict)

    return pd.DataFrame(rows)


def fva_rae(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    nlags: int = 1,
    baseline_type: Literal["naive", "moving_average"] = "naive",
    window_size: int = 5,
) -> float:
    """Calculate Relative Absolute Error (RAE) to evaluate Forecast Value Added.

    Parameters
    ----------
    y_true : Union[np.ndarray, List[float]]
        Ground-truth target values ordered chronologically.
    y_pred : Union[np.ndarray, List[float]]
        Predictions from the forecasting model to be evaluated.
    nlags : int, default=1
        The operational lead time or shift period (e.g., nlags=3 means decisions
        are made 3 periods prior).
    baseline_type : {"naive", "moving_average"}, default="naive"
        Type of baseline forecast to compare against.
        - "naive": Uses the value at lag-`nlags` (y[t - nlags]).
        - "moving_average": Uses trailing moving average ending at (t - nlags).
    window_size : int, default=5
        Window size for the moving average baseline (used when baseline_type="moving_average").

    Returns
    -------
    float
        The RAE value (MAE_model / MAE_baseline).
        - RAE < 1.0: Model adds value (FVA is positive).
        - RAE > 1.0: Model destroys value compared to baseline.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("All inputs must be 1-dimensional arrays.")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if nlags >= len(y_true):
        raise ValueError(
            "Number of lags cannot be greater than or equal to array length."
        )

    if nlags < 1:
        raise ValueError("Number of lags must be a positive integer >= 1.")

    y_true_eval = y_true[nlags:]
    y_pred_eval = y_pred[nlags:]

    if baseline_type == "naive":
        y_baseline_eval = y_true[:-nlags]

    elif baseline_type == "moving_average":
        if window_size < 2:
            raise ValueError("window_size must be >= 2 for moving_average baseline.")

        ma_series = rolling_window(y_true, window_size=window_size, func=np.mean)
        y_baseline_eval = ma_series[:-nlags]

    else:
        raise ValueError(
            f"Invalid baseline_type '{baseline_type}'. "
            "Supported options are 'naive' or 'moving_average'."
        )

    mae_model = np.mean(np.abs(y_true_eval - y_pred_eval))
    mae_baseline = np.mean(np.abs(y_true_eval - y_baseline_eval))

    if mae_baseline == 0:
        return np.nan if mae_model > 0 else 1.0

    return float(mae_model / mae_baseline)
