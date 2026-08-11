# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

from typing import Union
import numpy as np
from sklearn.base import ClassifierMixin

from tinyshift.utils.imports import requires_extra


@requires_extra("plot")
def efficiency_curve(
    clf: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    fig_type=None,
    width=800,
    height=400,
):
    """
    Compute and plot the efficiency and validity curves for a conformal classifier.

    Evaluates the model's statistical calibration (validity) and empirical
    usefulness (efficiency) across a predefined range of significance levels
    (alpha). This visualization helps diagnose under- or over-conservatism and
    identifies the optimal operating alpha for the classifier.

    Parameters
    ----------
    clf : object
        The conformal classifier model. Must implement a `predict_set(X, alpha)`
        method that returns a binary matrix of prediction sets.
    X : array-like of shape (n_samples, n_features)
        Input data features used for evaluation.
    y : array-like of shape (n_samples,) or (n_samples, 1)
        True target labels for the input data.
    fig_type : str, optional
        Format to automatically save or render the figure (e.g., 'png', 'svg').
        If None, the interactive plot is displayed in the browser.
    width : int, default=800
        The width of the generated Plotly figure in pixels.
    height : int, default=400
        The height of the generated Plotly figure in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        An interactive Plotly figure object containing the Validity, Efficiency,
        and Theoretical Coverage curves.

    Raises
    ------
    AttributeError
        If the provided `clf` object does not have a `predict_set` method.
    ValueError
        If the shapes of `X` and `y` are incompatible.

    Notes
    -----
    - **Validity** measures empirical coverage, defined as the proportion of
    samples where the true label is included in the prediction set. A valid
    model stays on or above the :math:`1 - \alpha` line.
    - **Efficiency** is quantified here as the *singleton rate* (the fraction of
    prediction sets containing exactly one class). Higher values indicate more
    informative and precise predictions.
    - A drop in efficiency at high alpha levels typically indicates the generation
    of empty prediction sets, as the model drops classes to meet the high
    permitted error budget.

    References
    ----------
    .. Angelopoulos, A. N., & Replinger, S. (2021). A gentle introduction to
    conformal prediction and distribution-free uncertainty quantification.
    arXiv preprint arXiv:2107.07511.
    .. Shafer, G., & Vovk, V. (2008). A tutorial on conformal prediction.
    Journal of Machine Learning Research, 9(3).
    """
    import plotly.graph_objects as go

    def get_error_metrics(clf, X: np.ndarray, y: np.ndarray) -> tuple:
        error_rate = np.asarray(
            [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.01]
        )
        efficiency_rate = np.zeros(error_rate.shape)
        validity_rate = np.zeros(error_rate.shape)

        y_flat = y.flatten().astype(int)
        n_samples = len(X)

        for i, error in enumerate(error_rate):
            predict_set = clf.predict_set(X, alpha=error)
            set_sizes = predict_set.sum(axis=1)
            efficiency_rate[i] = np.sum(set_sizes == 1) / n_samples
            covered = predict_set[np.arange(n_samples), y_flat]
            validity_rate[i] = np.mean(covered)

        return error_rate, efficiency_rate, validity_rate

    error_rate, efficiency_rate, validity_rate = get_error_metrics(clf, X, y)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=error_rate,
            y=efficiency_rate,
            mode="lines+markers",
            name="Efficiency (Singleton Rate)",
            line=dict(color="darkblue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=error_rate,
            y=validity_rate,
            mode="lines+markers",
            name="Validity (Empirical Coverage)",
            line=dict(color="orange"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=error_rate,
            y=1 - error_rate,
            mode="lines",
            name="Theoretical Coverage (1 - alpha)",
            line=dict(color="grey", dash="dash"),
        )
    )

    fig.update_layout(
        title="Efficiency & Validity Curve",
        xaxis_title="Significance Level (Alpha / Error Rate)",
        yaxis_title="Score / Rate",
        legend=dict(title="Metric"),
        width=width,
        height=height,
        hovermode="x",
    )
    fig.update_traces(hovertemplate="%{y:.2f}")

    return fig.show(fig_type)


@requires_extra("plot")
def reliability_curve(
    clf: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "Model",
    n_bins=15,
    fig_type=None,
    width=600,
    height=800,
):
    """
    Generates a reliability curve (calibration curve) for a binary classifier with calibration metrics summary.

    The reliability curve plots the true probability of the positive class against
    the mean predicted probability in each bin. A perfectly calibrated classifier
    would follow the diagonal line. The function also displays calibration metrics
    including Brier Score, Mean Calibration Error, and Maximum Calibration Error.

    Parameters
    -----------
    clf : ClassifierMixin
        The trained classifier model that implements predict_proba method.
    X : np.ndarray
        Input feature data for evaluation.
    y : np.ndarray
        True binary labels (0 or 1).
    model_name : str, optional
        Name to display in the legend (default: "Model").
    n_bins : int, optional
        Number of bins for the reliability curve (default: 15).
        Must be at least 2.
    fig_type : str, optional
        Display type for the figure (particularly useful in Jupyter notebooks).
        Common options: None (default), 'png', 'svg', 'browser', or other
        Plotly-supported renderers. (default: None)
    width : int, optional
        Figure width in pixels (default: 600).
    height : int, optional
        Figure height in pixels (default: 800).

    Returns
    --------
    None
        Displays the reliability curve plot with calibration metrics summary directly.

    Notes
    ------
    - Uses quantile strategy for binning to ensure equal-sized bins
    - The diagonal line represents perfect calibration
    - Deviations from the diagonal indicate miscalibration
    - **Brier Score**: Measures the mean squared difference between predicted
      probabilities and actual outcomes (lower is better, ranges 0-1)
    - **Mean Calibration Error**: Average absolute difference between predicted
      and true probabilities across bins
    - **Max Calibration Error**: Maximum absolute difference between predicted
      and true probabilities across bins

    Raises
    ------
    ValueError
        If classifier doesn't implement predict_proba method, if not binary
        classification, or if n_bins < 2.
    """
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    if not hasattr(clf, "predict_proba"):
        raise ValueError("The classifier must implement the 'predict_proba' method.")
    if len(np.unique(y)) != 2:
        raise ValueError("This function only supports binary classification.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    y_prob = clf.predict_proba(X)[:, 1]

    v_prob_true, v_prob_pred = calibration_curve(
        y, y_prob, n_bins=n_bins, strategy="quantile"
    )

    brier_score = brier_score_loss(y, y_prob)
    calibration_error = np.mean(np.abs(v_prob_true - v_prob_pred))
    max_calibration_error = np.max(np.abs(v_prob_true - v_prob_pred))

    fig = sp.make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Reliability Curve", "Calibration Summary"],
    )

    fig.add_trace(
        go.Scatter(
            x=v_prob_pred,
            y=v_prob_true,
            mode="lines+markers",
            name=model_name,
            line=dict(color="#1f77b4"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfectly calibrated",
            line=dict(dash="dash", color="grey"),
        ),
        row=1,
        col=1,
    )

    fig.update_xaxes(title_text="Mean predicted probability", row=1, col=1)
    fig.update_yaxes(title_text="Fraction of positives", row=1, col=1)

    summary_text = "<br>".join(
        [
            f"<b>Brier Score</b>: {brier_score:.4f}",
            f"<b>Mean Cal. Error</b>: {calibration_error:.4f}",
            f"<b>Max Cal. Error</b>: {max_calibration_error:.4f}",
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            text=[summary_text],
            mode="text",
            showlegend=False,
            textfont=dict(size=12),
            hoverinfo="skip",
            name="Summary",
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(visible=False, row=2, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)

    fig.update_layout(
        title="Reliability Curve with Calibration Metrics",
        width=width,
        height=height,
        showlegend=True,
    )

    return fig.show(fig_type)


@requires_extra("plot")
def beta_confidence_analysis(alpha, beta_param, fig_type=None):
    """
    Plot the Beta Probability Density Function (PDF) with filled area under the curve.

    This function creates an interactive visualization of the Beta distribution's PDF to assess
    model reliability. The distribution represents the balance between
    successes (alpha) and failures (beta) to help evaluate deployment confidence.

    Parameters
    -----------
    alpha : int or float
        The alpha (α) parameter representing model successes/correct predictions. Must be positive.
    beta_param : int or float
        The beta (β) parameter representing model failures/incorrect predictions. Must be positive.
    fig_type : str, optional
        Display type for the figure (particularly useful in Jupyter notebooks).
        Common options: None (default), 'notebook', or other Plotly-supported renderers.
        (default: None)

    Returns
    --------
    None
        Displays the Beta PDF plot directly.

    Notes
    ------
    - The Beta distribution is defined on the interval [0, 1]
    - Use the distribution shape and position to inform
      deployment decisions and risk assessment

    Raises
    -------
    ValueError
        If alpha or beta_param are not positive values.
    """
    import plotly.graph_objects as go
    from scipy.stats import beta

    if alpha <= 0 or beta_param <= 0:
        raise ValueError("Alpha and Beta parameters must be positive.")

    X = np.linspace(0, 1, 1000)
    y_pdf = beta.pdf(X, alpha, beta_param)

    fill_indices = (X >= 0) & (X <= 1)
    x_fill = X[fill_indices]
    y_pdf_fill = y_pdf[fill_indices]

    trace_pdf = go.Scatter(
        x=X, y=y_pdf, mode="lines", name=f"Beta PDF(α={alpha}, β={beta_param})"
    )

    trace_fill = go.Scatter(
        x=x_fill,
        y=y_pdf_fill,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(255,0,0,0.2)",
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False,
    )

    layout = go.Layout(
        title=f"Beta PDF (α={alpha}, β={beta_param})",
        xaxis=dict(
            title="Success Probability (x)",
            range=[
                min(x_fill[np.nonzero(y_pdf_fill)]),
                max(x_fill[np.nonzero(y_pdf_fill)]),
            ],
        ),
        yaxis=dict(title="Probability Density Function (PDF)"),
        width=800,
        height=450,
        hovermode="x unified",
    )

    fig = go.Figure(data=[trace_pdf, trace_fill], layout=layout)

    return fig.show(renderer=fig_type)


@requires_extra("plot")
def confusion_matrix(
    clf: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    fig_type=None,
    percentage_by_class=True,
):
    """
    Generate an annotated confusion matrix heatmap for a binary classifier.

    This function creates an interactive visualization of the confusion matrix,
    displaying the counts of true positives, true negatives, false positives,
    and false negatives. The matrix can show percentages either by class or
    as overall percentages of the total predictions.

    Parameters
    -----------
    clf : ClassifierMixin
        The trained classifier model that implements predict method.
    X : np.ndarray
        Input feature data for evaluation.
    y : np.ndarray
        True binary labels (0 or 1).
    fig_type : str, optional
        Display type for the figure (particularly useful in Jupyter notebooks).
        Common options: None (default), 'notebook', or other Plotly-supported renderers.
        (default: None)
    percentage_by_class : bool, optional
        If True, displays percentages by class (each column sums to 100%).
        If False, displays overall percentages (all cells sum to 100%).
        (default: True)

    Returns
    --------
    None
        Displays the confusion matrix heatmap directly.

    Notes
    ------
    - The matrix follows the format: rows represent true labels, columns represent predicted labels
    - TN: True Negative, FP: False Positive, FN: False Negative, TP: True Positive
    - Color intensity represents the magnitude of values in each cell
    """
    import plotly.figure_factory as ff
    from sklearn import metrics

    y_pred = clf.predict(X)
    tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred).ravel()
    labels = np.array([["FN", "TN"], ["TP", "FP"]])
    cm = np.array([[fn, tn], [tp, fp]])

    if percentage_by_class:
        total = cm.sum(axis=0)
        percentage = cm / total * 100
    else:
        percentage = cm / np.sum(cm) * 100

    annotation_text = np.empty_like(percentage, dtype="U10")

    for i in range(percentage.shape[0]):
        for j in range(percentage.shape[1]):
            annotation_text[i, j] = f"{labels[i, j]} {percentage[i, j]:.2f}"

    fig = ff.create_annotated_heatmap(
        cm,
        x=["Positive", "Negative"],
        y=["Negative", "Positive"],
        colorscale="Blues",
        hoverinfo="z",
        annotation_text=annotation_text,
    )

    fig.update_layout(width=400, height=400, title="Confusion Matrix")
    return fig.show(fig_type)


@requires_extra("plot")
def score_distribution(
    clf: ClassifierMixin,
    X: np.ndarray,
    nbins=15,
    fig_type=None,
):
    """
    Generate a histogram showing the distribution of predicted scores for a classifier.

    This function creates an interactive visualization displaying the distribution of
    predicted probabilities from a binary classifier. The histogram helps identify
    patterns in model predictions and potential calibration issues.

    Parameters
    ----------
    clf : ClassifierMixin
        The trained classifier model that implements predict_proba method.
    X : np.ndarray
        Input feature data for evaluation.
    nbins : int, optional
        Number of bins for the histogram (default: 15).
    fig_type : str, optional
        Display type for the figure (particularly useful in Jupyter notebooks).
        Common options: None (default), 'notebook', or other Plotly-supported renderers.
        (default: None)

    Returns
    -------
    None
        Displays the histogram plot directly.

    Notes
    -----
    - Shows the distribution of predicted probabilities for the positive class
    - Useful for understanding model confidence patterns
    - Can help identify potential calibration issues
    - Well-calibrated models typically show varied probability distributions
    """
    import plotly.express as px

    y_prob = clf.predict_proba(X)[:, 1]
    fig = px.histogram(y_prob, nbins=nbins)
    fig.update_layout(
        title="Histogram of Predicted Scores",
        xaxis_title="Predicted Scores",
        yaxis_title="Count",
        legend_title="Modelos",
        autosize=False,
    )
    fig.update_layout(hovermode="x")
    fig.update_traces(hovertemplate="%{y}")
    fig.update_layout(showlegend=False)
    return fig.show(fig_type)
