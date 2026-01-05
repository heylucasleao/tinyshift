# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

import numpy as np
import plotly.graph_objects as go
from statsmodels.stats.power import TTestIndPower
from typing import Optional, Union


def power_curve(
    effect_size: float,
    alpha: float = 0.05,
    power_target: float = 0.80,
    height: int = 500,
    width: int = 600,
    fig_type: Optional[str] = None,
):
    """
    Creates an interactive Power Analysis plot showing statistical power versus sample size.

    This function generates a comprehensive visualization to assess the relationship
    between sample size and statistical power for a two-sample t-test. The plot
    displays the power curve, highlights the required sample size to achieve the
    target power, and provides insights about the effect size magnitude.

    Parameters
    ----------
    effect_size : float
        Cohen's d effect size to be detected in the analysis.
        Must be greater than zero.
    alpha : float, default=0.05
        Significance level (Type I error probability).
        Must be between 0 and 1.
    power_target : float, default=0.80
        Target statistical power level.
        Must be between 0 and 1.
    height : int, default=500
        Figure height in pixels.
    width : int, default=600
        Figure width in pixels.
    fig_type : str, optional
        Plotly figure output type. Passed to `fig.show()`.
        E.g.: 'png', 'html', 'svg', 'json'.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Returns the Plotly Figure object if `fig_type` is `None` or the result
        of the `fig.show(fig_type)` call.

    Raises
    ------
    ValueError
        If effect_size is not positive, alpha is not between 0 and 1,
        or power_target is not between 0 and 1.

    Notes
    -----
    The plot includes confidence intervals and highlights the optimal sample size
    required to achieve the target power level for the specified effect size.

    """

    if effect_size <= 0:
        raise ValueError("Effect size must be greater than zero.")
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1.")
    if not (0 < power_target < 1):
        raise ValueError("Power target must be between 0 and 1.")

    analysis = TTestIndPower()
    n_required = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power_target,
        ratio=1.0,
        alternative="two-sided",
    )

    sample_sizes = np.arange(10, int(n_required * 1.8) + 1, 1)

    powers = analysis.power(
        effect_size=effect_size,
        nobs1=sample_sizes,
        alpha=alpha,
        ratio=1.0,
        alternative="two-sided",
    )

    n_optimal = int(np.ceil(n_required))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=powers,
            mode="lines",
            name="Statistical Power",
            line=dict(color="#2E86AB", width=3),
            hovertemplate="<b>Sample Size:</b> %{x}<br><b>Power:</b> %{y:.2%}<extra></extra>",
        )
    )
    fig.add_hline(
        y=power_target,
        line_dash="dash",
        line_color="grey",
        line_width=2,
        annotation_text=f"Target Power ({power_target:.0%})",
        annotation_position="bottom right",
    )
    fig.add_vline(
        x=n_optimal,
        line_dash="dash",
        line_color="grey",
        line_width=2,
        annotation_text=f"n = {n_optimal}",
        annotation_position="top left",
    )
    fig.add_trace(
        go.Scatter(
            x=[n_optimal],
            y=[power_target],
            mode="markers",
            name=f"Optimal Point (n={n_optimal})",
            marker=dict(
                size=6,
                color="darkred",
                symbol="circle",
                line=dict(width=2, color="darkred"),
            ),
            hovertemplate=f"<b>Optimal Sample Size:</b> {n_optimal}<br><b>Target Power:</b> {power_target:.0%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Power Analysis<br><sub>Effect Size (d) = {effect_size:.3f}, α = {alpha:.3f}</sub>",
            x=0.5,
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Sample Size (n per group)",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            gridcolor="lightgray",
        ),
        yaxis=dict(
            title="Statistical Power",
            title_font=dict(size=14),
            tickformat=".0%",
            range=[0, 1.05],
            tickfont=dict(size=12),
            gridcolor="lightgray",
        ),
        showlegend=True,
        legend=dict(orientation="v", yanchor="bottom", y=0.02, xanchor="right", x=0.98),
        hovermode="x unified",
        height=height,
        width=width,
    )

    effect_interpretation = ""
    if effect_size < 0.2:
        effect_interpretation = "Very Small"
    elif effect_size < 0.5:
        effect_interpretation = "Small"
    elif effect_size < 0.8:
        effect_interpretation = "Medium"
    else:
        effect_interpretation = "Large"

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Effect Size: {effect_interpretation}",
        showarrow=False,
        font=dict(size=12, color="black"),
        bordercolor="gray",
        borderwidth=1,
    )

    return fig.show(fig_type)


def power_vs_allocation(
    effect_size: float,
    sample_size: int,
    alpha: float = 0.05,
    height: int = 500,
    width: int = 600,
    fig_type: str = None,
):
    """
    Creates an interactive plot showing Statistical Power as a function of treatment allocation.

    This function generates a comprehensive visualization to assess how the proportion
    of customers allocated to the treatment group affects the statistical power of
    an A/B test, keeping the total sample size fixed. The plot demonstrates that
    power is maximized when allocation is balanced (50/50 split).

    Parameters
    ----------
    effect_size : float
        Cohen's d effect size to be detected in the analysis.
        Must be greater than zero.
    sample_size : int
        Total sample size (control + treatment groups combined).
        Must be positive.
    alpha : float, default=0.05
        Significance level (Type I error probability).
        Must be between 0 and 1.
    height : int, default=500
        Figure height in pixels.
    width : int, default=600
        Figure width in pixels.
    fig_type : str, optional
        Plotly figure output type. Passed to `fig.show()`.
        E.g.: 'png', 'html', 'svg', 'json'.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Returns the Plotly Figure object if `fig_type` is `None` or the result
        of the `fig.show(fig_type)` call.

    Raises
    ------
    ValueError
        If effect_size is not positive, sample_size is not positive,
        or alpha is not between 0 and 1.

    Notes
    -----
    The plot shows that statistical power is maximized at 50% treatment allocation
    (equal group sizes) and decreases as allocation becomes more imbalanced.
    """

    if sample_size <= 0:
        raise ValueError("Total sample size must be positive.")
    if effect_size <= 0:
        raise ValueError("Effect size must be greater than zero.")
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1.")

    analysis = TTestIndPower()

    treatment_proportions = np.array(
        [0.05, 0.15, 0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75, 0.85, 0.95]
    )

    powers = []

    for p_treat in treatment_proportions:

        if p_treat == 0.5:
            current_ratio = 1.0
        else:
            current_ratio = p_treat / (1.0 - p_treat)

        n_control = sample_size * (1.0 - p_treat)

        if n_control < 2:
            powers.append(0)
            continue

        calculated_power = analysis.power(
            effect_size=effect_size,
            nobs1=n_control,
            alpha=alpha,
            ratio=current_ratio,
            alternative="two-sided",
        )
        powers.append(calculated_power)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=treatment_proportions,
            y=powers,
            mode="lines+markers",
            name="Statistical Power",
            line=dict(color="#2E86AB", width=3),
            marker=dict(size=8, color="#2E86AB"),
            hovertemplate="<b>Treatment Allocation:</b> %{x:.0%}<br><b>Power:</b> %{y:.2%}<extra></extra>",
        )
    )

    fig.add_vline(
        x=0.50,
        line_dash="dash",
        line_color="grey",
        line_width=2,
        annotation_text="50%",
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(
            text=f"Power vs. Treatment Allocation<br><sub>Total N = {sample_size}, Effect Size (d) = {effect_size:.3f}, α = {alpha:.3f}</sub>",
            x=0.5,
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Proportion of Allocation to Treatment",
            title_font=dict(size=14),
            tickformat=".0%",
            tickfont=dict(size=12),
            gridcolor="lightgray",
        ),
        yaxis=dict(
            title="Statistical Power",
            title_font=dict(size=14),
            tickformat=".0%",
            range=[0, 1.05],
            tickfont=dict(size=12),
            gridcolor="lightgray",
        ),
        showlegend=True,
        legend=dict(orientation="v", yanchor="bottom", y=0.02, xanchor="right", x=0.98),
        hovermode="x unified",
        height=height,
        width=width,
    )

    return fig.show(fig_type)
