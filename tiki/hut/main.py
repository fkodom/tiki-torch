import argparse
from math import ceil

import streamlit as st
import plotly.graph_objects as go

from tiki.hut.data import get_logs_data
from tiki.hut.config import figure_config


"""
# Tiki Hut

Your island headquarters and training visualization dashboard.
"""

parser = argparse.ArgumentParser()
parser.add_argument("logdir")
logdir = parser.parse_args().logdir

logs = get_logs_data(logdir)
log_names = tuple(log["name"] for log in logs)

if len(logs) > 0:
    page = st.selectbox("Select: ", ("Graphs", "Scalars", "Third"))

    if page == "Graphs":
        log_name = st.selectbox("Model:", log_names)
        for log in logs:
            if log["name"] == log_name and "graph" in log.keys():
                st.write(log["graph"])

    if page == "Scalars":
        all_scalars = [scalar for log in logs for scalar in log["history"].keys()]
        scalars = [x for x in set(all_scalars) if x not in ["epochs", "batches"]]
        show_legend = st.checkbox("Show legend", value=True)

        nrow = int(len(scalars) ** 0.5)
        ncol = ceil(len(scalars) / nrow)

        for scalar in scalars:
            fig = go.Figure()
            for log in logs:
                if "history" in log.keys():
                    history = log["history"]
                else:
                    continue

                if scalar in history.keys():
                    fig.add_trace(
                        go.Scatter(
                            x=history["epochs"],
                            y=history[scalar],
                            name=log["name"],
                            hovertext=[
                                f"{log['name']}: epoch={epoch}, {scalar}={x:.2e}"
                                for epoch, x in zip(history["epochs"], history[scalar])
                            ],
                            hoverinfo="text",
                        )
                    )
            fig.update_layout(
                **figure_config,
                showlegend=show_legend,
                title=scalar,
                xaxis=go.layout.XAxis(title="epoch"),
                yaxis=go.layout.YAxis(title=scalar),
            )
            st.write(fig)
else:
    """
    ### Found no logs to display.  
    
    First, train a model using the `TikiHut` callback.
    """
