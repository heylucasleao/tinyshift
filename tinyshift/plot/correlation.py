# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

from typing import Optional
import numpy as np
from sklearn.utils.validation import check_array

from tinyshift.utils.imports import requires_extra


@requires_extra("plots")
def corr_heatmap(
    X: np.ndarray,
    width: int = 1600,
    height: int = 1600,
    fig_type: Optional[str] = None,
):
    """
    Generate an interactive correlation heatmap for a DataFrame using Plotly Express.

    This function visualizes the correlation matrix between all numeric columns in the DataFrame,
    with values displayed on each cell and a diverging color scale (blue for negative, red for positive correlations).

    Parameters
    -----------
    df : np.ndarray
        A matrix containing numeric columns to compute correlations.
        Non-numeric columns will be automatically excluded.
    width : int, optional
        Width of the figure in pixels (default: 1600)
    height : int, optional
        Height of the figure in pixels (default: 1600)
    fig_type : str, optional
        Display type for the figure (particularly useful in Jupyter notebooks).
        Common options: None (default), 'notebook', or other Plotly-supported renderers.
        (default: None)

    Returns
    --------
    None
        Displays the heatmap directly.

    Examples
    --------
    >>> # Basic usage with default size
    >>> corr_heatmap(df)

    >>> # Custom size
    >>> corr_heatmap(df, width=1200, height=1200)

    >>> # For Jupyter notebook display
    >>> corr_heatmap(df, fig_type='notebook')

    Notes
    ------
    - The correlation matrix is computed using pandas.DataFrame.corr() (Pearson correlation)
    - The color scale ranges from -1 (perfect negative correlation) to +1 (perfect positive correlation)
    - Diagonal elements will always be 1 (perfect self-correlation)
    - Only numeric columns are included in the calculation (equivalent to numeric_only=True)
    """
    import plotly.express as px

    feature_names_in_ = getattr(X, "columns", None)
    X = check_array(X, ensure_2d=True, dtype=np.float64, copy=True)

    if feature_names_in_ is None:
        feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]

    corr = np.corrcoef(X, rowvar=False)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="equal",
        color_continuous_scale="RdBu_r",
        title="Heatmap of Correlation Coefficients",
        labels=dict(color="Correlation"),
        x=feature_names_in_,
        y=feature_names_in_,
    )

    fig.update_layout(
        width=width,
        height=height,
        xaxis_title="Features",
        yaxis_title="Features",
        title_x=0.5,
    )

    return fig.show(fig_type)
