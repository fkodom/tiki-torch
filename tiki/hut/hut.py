import argparse

import streamlit as st

from tiki.hut.data import load_logs_data
from tiki.hut.metrics import write_metrics
from tiki.hut.hyperparams import write_hyperparams
from tiki.hut.graphs import write_graphs
from tiki.hut.histogram import write_histogram


parser = argparse.ArgumentParser()
parser.add_argument("logdir")
logdir = parser.parse_args().logdir
logs = load_logs_data(logdir)

st.sidebar.markdown(
    """
    # Tiki Hut
    
    Your island headquarters and training visualization dashboard.
    """
)

refresh = st.sidebar.button("Refresh Data")
if refresh:
    st.caching.clear_cache()

st.sidebar.markdown("""### Navigation""")
page = st.sidebar.radio(
    "Section: ",
    ("Metrics", "Hyperparameters", "Computation Graphs", "Histograms")
)

if len(logs) > 0:
    if page == "Metrics":
        write_metrics(logs)
    elif page == "Hyperparameters":
        write_hyperparams(logs)
    elif page == "Computation Graphs":
        write_graphs(logs)
    elif page == "Histograms":
        write_histogram(logs)

else:
    """
    ### Found no logs to display
    """
