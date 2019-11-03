"""
metrics.py
----------
Methods for writing graphs of training metrics to Tiki-Hut
"""

from typing import Iterable, Dict

import streamlit as st
import plotly.graph_objects as go

from tiki.hut.config import figure_config


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = ["write_metrics"]


def _write_custom_plot(
    logs: Iterable[Dict], xlabel: str, ylabel: str, **config
) -> None:
    """Writes a `go.Scatter` plot with x- and y-axes specified by `xlabel` and
    `ylabel`.  A separate line is written for each log file.

    Parameters
    ----------
    logs: Iterable[dict]
        Iterable of training logs. Each is a dictionary of training information
    xlabel: str
        String specifying the metric to plot along the x-axis
    ylabel: str
        String specifying the metric to plot along the y-axis
    **config
        Additional keyword arguments for customizing the histogram figure.
    """
    fig = go.Figure()
    for log in logs:
        if "history" in log.keys() and ylabel in log["history"].keys():
            xlabels = log["history"][xlabel]
            ylabels = log["history"][ylabel]
            times = log["history"]["time"]

            fig.add_trace(
                go.Scatter(
                    x=xlabels,
                    y=ylabels,
                    name=log["name"],
                    hovertext=[
                        f"{log['name']}<br>"
                        f"{t.strftime('%d %b %y %H:%M:%S')}<br>"
                        f"{xlabel}={x:.2e}<br>"
                        f"{ylabel}={y:.2e}"
                        for x, y, t in zip(xlabels, ylabels, times)
                    ],
                    hoverinfo="text",
                )
            )
            fig.update_layout(
                **config,
                title=f"{ylabel}",
                xaxis=go.layout.XAxis(title=xlabel),
                yaxis=go.layout.YAxis(title=ylabel),
            )

    st.write(fig)


def _write_default_plots(logs: Iterable[Dict], tags: Iterable[str], **config) -> None:
    """For each metric specified in `tags`, writes a `go.Scatter` plot with
    "epochs" along the x-axis and the metric along the y-axis.

    Parameters
    ----------
    logs: Iterable[dict]
        Iterable of training logs. Each is a dictionary of training information
    tags: Iterable[str]
        Iterable of strings specifying which metrics to plot
    **config
        Additional keyword arguments for customizing the histogram figure.
    """
    for scalar in tags:
        _write_custom_plot(logs, "epochs", scalar, **config)


def _show_legend():
    showlegend = st.checkbox("Show legend", value=True)
    figure_config["showlegend"] = showlegend


def write_metrics(logs: Iterable[Dict]) -> None:
    """Provides users with (1) the option to display default plots or create
    their own custom plot and (2) the option to show/hide legends.  Then,
    writes `go.Scatter` plots to Tiki-Hut.

    Parameters
    ----------
    logs: Iterable[dict]
        Iterable of training logs. Each is a dictionary of training information
    """
    st.write("""# Metrics""")
    # for log in logs:
    #     if "history" not in log.keys():
    #         log["history"] = {}

    all_scalars = []
    for log in logs:
        if "history" in log.keys():
            all_scalars += list(log["history"].keys())
    # all_scalars = [scalar for log in logs for scalar in log["history"].keys()]
    all_scalars = sorted(list(set(all_scalars)))
    scalars = [x for x in all_scalars if x not in ["epochs", "batches", "time"]]
    custom_plot = st.checkbox("Custom plot", value=False)

    if custom_plot:
        st.write("""### Select Axes""")
        xlabel = st.selectbox("X:", ["-- Select --", *list(all_scalars)])
        ylabel = st.selectbox("Y:", ["-- Select --", *list(all_scalars)])
        _show_legend()
        _write_custom_plot(logs, xlabel, ylabel, **figure_config)
    else:
        _show_legend()
        _write_default_plots(logs, scalars, **figure_config)
