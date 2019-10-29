from typing import Iterable, Dict

import streamlit as st
import plotly.graph_objects as go


def get_custom_plot(logs: Iterable[Dict], xlabel: str, ylabel: str, **config):
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


def get_default_plots(logs: Iterable[Dict], scalars: Iterable[str], **config):
    for scalar in scalars:
        get_custom_plot(logs, "epochs", scalar, **config)
