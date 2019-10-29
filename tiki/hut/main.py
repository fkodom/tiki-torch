import argparse
from contextlib import redirect_stdout

import streamlit as st

from tiki.hut.data import get_logs_data
from tiki.hut.scalars import get_default_plots, get_custom_plot
from tiki.hut.hyperparams import get_hyperparams
from tiki.hut.config import figure_config


"""
# Tiki Hut

Your island headquarters and training visualization dashboard.
"""

parser = argparse.ArgumentParser()
parser.add_argument("logdir")
logdir = parser.parse_args().logdir

with redirect_stdout(None):
    logs = get_logs_data(logdir)
log_names = list(log["name"] for log in logs)

if len(logs) > 0:
    page = st.selectbox("Select: ", ("Metrics", "Hyperparams", "Graphs"))

    if page == "Metrics":
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
            """
            ### Select Axes
            """
            xlabel = st.selectbox("X:", ["-- Select --", *list(all_scalars)])
            ylabel = st.selectbox("Y:", ["-- Select --", *list(all_scalars)])
            get_custom_plot(logs, xlabel, ylabel, **figure_config)
        else:
            get_default_plots(logs, scalars, **figure_config)

    elif page == "Graphs":
        log_name = st.selectbox("Model:", log_names)
        for log in logs:
            if log["name"] == log_name and "graph" in log.keys():
                st.write(log["graph"])

    elif page == "Hyperparams":
        get_hyperparams(logs)

    # TODO: Histograms
    # Use Plotly 3D Filled Line plots

else:
    """
    ### Found no logs to display.  
    
    First, train a model using the `TikiHut` callback.
    """
