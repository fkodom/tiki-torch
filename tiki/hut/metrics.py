from typing import Iterable, Dict

import streamlit as st
import plotly.graph_objects as go

from tiki.hut.config import figure_config


def _write_custom_plot(logs: Iterable[Dict], xlabel: str, ylabel: str, **config):
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
                        f"{t.strftime('%Y %b %d %H:%M:%S')}<br>"
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


def _write_default_plots(logs: Iterable[Dict], scalars: Iterable[str], **config):
    for scalar in scalars:
        _write_custom_plot(logs, "epochs", scalar, **config)


def write_metrics(logs: Iterable[Dict]):
    for log in logs:
        if "history" not in log.keys():
            log["history"] = {}

    all_scalars = [scalar for log in logs for scalar in log["history"].keys()]
    all_scalars = sorted(list(set(all_scalars)))
    scalars = [x for x in all_scalars if x not in ["epochs", "batches", "time"]]
    custom_plot = st.checkbox("Custom plot", value=False)
    showlegend = st.checkbox("Show legend", value=True)
    figure_config["showlegend"] = showlegend

    if custom_plot:
        st.write(
            """
            ### Select Axes
            """
        )
        xlabel = st.selectbox("X:", ["-- Select --", *list(all_scalars)])
        ylabel = st.selectbox("Y:", ["-- Select --", *list(all_scalars)])
        _write_custom_plot(logs, xlabel, ylabel, **figure_config)
    else:
        _write_default_plots(logs, scalars, **figure_config)
