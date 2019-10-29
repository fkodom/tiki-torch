import argparse

import streamlit as st

from tiki.hut.data import load_logs_data
from tiki.hut.metrics import write_metrics
from tiki.hut.hyperparams import write_hyperparams
from tiki.hut.graphs import write_graphs
from tiki.hut.histogram import write_histogram


"""
# Tiki Hut

Your island headquarters and training visualization dashboard.
"""

parser = argparse.ArgumentParser()
parser.add_argument("logdir")
logdir = parser.parse_args().logdir
logs = load_logs_data(logdir)

if len(logs) > 0:
    page = st.selectbox("Select: ", ("Metrics", "Hyperparams", "Graphs", "Histograms"))

    if page == "Metrics":
        write_metrics(logs)
    elif page == "Hyperparams":
        write_hyperparams(logs)
    elif page == "Graphs":
        write_graphs(logs)
    elif page == "Histograms":
        write_histogram(logs)

else:
    """
    ### Found no logs to display
    """
