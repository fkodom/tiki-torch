from typing import Iterable, Sized, List, Dict

import plotly.graph_objects as go
import panel as pn
from panel.widgets import DiscreteSlider


plotly_fig_kwargs = {
    "autosize": False,
    "width": 600,
    "height": 450,
    "margin": {"l": 30, "r": 30, "t": 40, "b": 30}
}


def fpr_tpr_plots(
    algorithm_logs: Dict,
    markers: Iterable = (),
    lines: Iterable = ()
):
    markers = list(markers)
    lines = list(lines)
    num_algorithms = len(algorithm_logs.keys())

    while len(markers) < num_algorithms:
        markers.append(None)
    while len(lines) < num_algorithms:
        lines.append(None)

    return [
        go.Scatter(
            x=log["fp_rates"],
            y=log["tp_rates"],
            marker=marker,
            line=line,
            hovertext=[
                f"FPR={x:.2E}, TPR={y:.2E}, Thresh={t:.2E}" for x, y, t in
                zip(log["fp_rates"], log["tp_rates"], log["thresholds"])
            ],
            hoverinfo="text",
            name=algorithm
        )
        for (algorithm, log), marker, line in zip(algorithm_logs.items(), markers, lines)
    ]


def precision_recall_plots(
        algorithm_logs: Dict,
        markers: Iterable = (),
        lines: Iterable = ()
):
    markers = list(markers)
    lines = list(lines)
    num_algorithms = len(algorithm_logs.keys())

    while len(markers) < num_algorithms:
        markers.append(None)
    while len(lines) < num_algorithms:
        lines.append(None)

    return [
        go.Scatter(
            x=log["precisions"],
            y=log["recalls"],
            marker=marker,
            line=line,
            hovertext=[
                f"P={x:.2E}, R={y:.2E}, Thresh={t:.2E}" for x, y, t in
                zip(log["precisions"], log["recalls"], log["thresholds"])
            ],
            hoverinfo="text",
            name=algorithm,
            showlegend=False
        )
        for (algorithm, log), marker, line in zip(algorithm_logs.items(), markers, lines)
    ]


def fpr_tpr(
    algorithm_logs: Dict,
    markers: Iterable = (),
    lines: Iterable = ()
):
    fig = go.Figure()
    plots = fpr_tpr_plots(
        algorithm_logs,
        markers=markers,
        lines=lines,
    )
    [fig.add_trace(plot) for plot in plots]
    fig.update_layout(
        **plotly_fig_kwargs,
        title=go.layout.Title(text="FP Rate vs TP Rate"),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(text=f"FP Rate")
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(text="TP Rate")
        ),
        legend=go.layout.Legend(x=0.63, y=0.05)
    )

    return fig


def precision_recall(
    algorithm_logs: Dict,
    markers: Iterable = (),
    lines: Iterable = ()
):
    fig = go.Figure()
    plots = precision_recall_plots(
        algorithm_logs,
        markers=markers,
        lines=lines,
    )
    [fig.add_trace(plot) for plot in plots]
    fig.update_layout(
        **plotly_fig_kwargs,
        title=go.layout.Title(text="Precision vs Recall"),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(text=f"Precision")
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(text="Recall")
        )
    )

    return fig
