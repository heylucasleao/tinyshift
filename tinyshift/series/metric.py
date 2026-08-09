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
    errors = (
        df[models]
        .sub(df[target_col], axis=0)
        .abs()
        .groupby(df[id_col], observed=True)
        .sum()
    )

    demand = df[target_col].abs().groupby(df[id_col], observed=True).sum().values

    res = errors.div(demand, axis=0).mul(100.0).reset_index()
    res[models] = np.where(
        demand[:, None] == 0, np.where(errors == 0, 0.0, np.nan), res[models]
    )

    res.insert(1, "metric", "wape")
    return res


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
    errors = (
        df[models].sub(df[target_col], axis=0).groupby(df[id_col], observed=True).sum()
    )

    demand = df[target_col].groupby(df[id_col], observed=True).sum().values
    res = errors.div(demand, axis=0).mul(100.0).reset_index()
    res[models] = np.where(
        demand[:, None] == 0, np.where(errors == 0, 0.0, np.nan), res[models]
    )

    res.insert(1, "metric", "pbias")
    return res


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
    References
    ----------
    - Vandeput, N. (2025). Forecasting Variability: Causes, Solutions, and
      Why It (doesn't) Matter. SupChains.
      Available at: https://www.supchains.com/
    - Vandeput, N. (2021). Data Science for Supply Chain Forecasting (2nd ed.).
      CRC Press.
    """

    df_wape = wape(df=df, models=models, id_col=id_col, target_col=target_col)
    df_pbias = pbias(df=df, models=models, id_col=id_col, target_col=target_col)
    df_score = df_wape[[id_col]].copy()
    df_score["metric"] = "score"
    df_score[models] = df_wape[models] + df_pbias[models].abs()

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

    References
    ----------
    - Morlidge, S. (2014). The Little Book of Business Forecasting:
        A Practical Guide to Measuring and Improving Forecast Performance.
        Business Forecasting Press.
    - Gilliland, M. (2010). The Business Forecasting Deal: Exposing the
        Myths, Eliminating the Waste, and Practicing the Realities. John Wiley & Sons.
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

    References
    ----------
    - Morlidge, S. (2014). The Little Book of Business Forecasting:
        A Practical Guide to Measuring and Improving Forecast Performance.
        Business Forecasting Press.
    - Gilliland, M. (2010). The Business Forecasting Deal: Exposing the
        Myths, Eliminating the Waste, and Practicing the Realities. John Wiley & Sons.
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


def forecast_instability(
    df: pd.DataFrame,
    models: List[str],
    ds_col: str = "ds",
    id_col: str = "unique_id",
    **kwargs,
) -> pd.DataFrame:
    """Calculate Forecasting Instability Error across periods for multiple models.

    Measures period-over-period forecast instability by evaluating all consecutive
    time step pairs (ds_t-1 vs ds_t) across the entire history of each series.
    It reuses the composite `score` function (WAPE + |PBias|) calculated with the
    previous period (ds_t-1) as target, applying a scale factor to align the
    denominator with the joint average volume across all paired periods:
    (sum(|F_prev - F_curr|) + |sum(F_prev - F_curr)|) / (0.5 * (sum(F_prev) + sum(F_curr))).

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation DataFrame containing series identifiers, date/period column, and model forecasts.
    models : List[str]
        List of column names corresponding to the forecast models to evaluate.
    ds_col : str, default="ds"
        Column name identifying time periods or forecast dates ordered chronologically.
    id_col : str, default="unique_id"
        Column name identifying unique series or group identifiers.

    Returns
    -------
    pd.DataFrame
        A DataFrame formatted with `id_col`, `metric` label ('forecast_instability'), and
        columns for each evaluated model containing their respective instability values.

    Notes
    -----
    Interpretation & Aggregation:
    - Measures forecast revision magnitude and operational instability across consecutive periods.
    - Expressed as a percentage (%):
      * 0.0%: Perfectly stable forecast (zero revisions across periods).
      * Lower values indicate higher stability.
      * Stability percentage can be derived as: Stability = 100% - Forecast Instability.
    - Unbounded upper limit: Can exceed 100% when forecast revisions are aggressive.
    - Aggregation Mechanics per Series (`id_col`):
      * For a series with N time steps, creates N-1 consecutive pairs (F_{t-1}, F_t).
      * Numerator: Sums absolute differences `sum(|F_{t-1} - F_t|)` and net directional drift
        `|sum(F_{t-1} - F_t)|` across ALL N-1 paired periods of the series.
      * Denominator: Calculates total volume of prior periods `sum(F_{t-1})` and current periods
        `sum(F_t)`, taking their joint average `0.5 * (sum(F_{t-1}) + sum(F_t))`.
      * Internally reuses `score()` treating `F_{t-1}` as target, then applies the volume
        scaling factor `sum(F_prev) / (0.5 * (sum(F_prev) + sum(F_curr)))`.
    - Example: An instability of 15.0% means period-over-period forecast adjustments
      account for 15% of the average projected volume across all consecutive periods.

    References
    ----------
    - Vandeput, N. (2025). Forecasting Variability: Causes, Solutions, and
        Why It (doesn't) Matter. SupChains.
        Available at: https://www.supchains.com/
    - Vandeput, N. (2021). Data Science for Supply Chain Forecasting (2nd ed.).
        CRC Press.
    """

    def _prepare_paired_data(
        df_in: pd.DataFrame,
        models_list: List[str],
        id_column: str,
        ds_column: str,
    ) -> pd.DataFrame:
        """Sort data, apply group shift, and return paired consecutive forecasts."""
        df_sorted = df_in.sort_values([id_column, ds_column])

        df_curr = df_sorted[[id_column] + models_list].copy()
        df_prev = (
            df_sorted.groupby(id_column, observed=True)[models_list]
            .shift(1)
            .add_suffix("_prev")
        )

        paired_df = pd.concat([df_curr, df_prev], axis=1).dropna()

        return paired_df

    def _compute_metrics_and_consolidate(
        paired_df: pd.DataFrame,
        models_list: List[str],
        id_column: str,
    ) -> pd.DataFrame:
        """Compute base score, scale by joint average volume, and consolidate model results."""
        scores_dict = {}

        for model in models_list:
            target_col_name = f"{model}_prev"
            df_eval = pd.DataFrame(
                {
                    id_column: paired_df[id_column],
                    "target": paired_df[target_col_name],
                    model: paired_df[model],
                }
            )

            df_raw_score = score(
                df=df_eval,
                models=[model],
                id_col=id_column,
                target_col="target",
            )

            sum_prev = df_eval.groupby(id_column, observed=True)["target"].sum()
            sum_curr = df_eval.groupby(id_column, observed=True)[model].sum()
            avg_volume = 0.5 * (sum_prev + sum_curr)

            scale_factor = np.where(avg_volume == 0, 1.0, sum_prev / avg_volume)
            scaled_score = df_raw_score[model] * scale_factor
            scores_dict[model] = pd.Series(
                scaled_score.values, index=df_raw_score[id_column]
            )

        results_df = pd.DataFrame(scores_dict).reset_index()
        results_df.insert(1, "metric", "forecast_instability")

        return results_df

    def _ensure_all_unique_ids(
        results_df: pd.DataFrame,
        original_df: pd.DataFrame,
        models_list: List[str],
        id_column: str,
    ) -> pd.DataFrame:
        """Ensure all original unique IDs are present in the final results DataFrame."""
        all_ids_df = pd.DataFrame({id_column: original_df[id_column].unique()})

        if results_df.empty:
            res = all_ids_df.assign(metric="forecast_instability")
            for model in models_list:
                res[model] = np.nan
            return res

        res = all_ids_df.merge(results_df, on=id_column, how="left")
        res["metric"] = "forecast_instability"
        return res

    paired = _prepare_paired_data(
        df_in=df, models_list=models, id_column=id_col, ds_column=ds_col
    )

    if paired.empty:
        return _ensure_all_unique_ids(
            results_df=pd.DataFrame(),
            original_df=df,
            models_list=models,
            id_column=id_col,
        )

    res = _compute_metrics_and_consolidate(
        paired_df=paired, models_list=models, id_column=id_col
    )

    res = _ensure_all_unique_ids(
        results_df=res, original_df=df, models_list=models, id_column=id_col
    )

    return res
