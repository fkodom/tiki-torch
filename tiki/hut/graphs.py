"""
graphs.py
---------
Method(s) for writing computation graphs to Tiki-Hut
"""

from typing import Iterable, Dict

import streamlit as st


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = ["write_graphs"]


def write_graphs(logs: Iterable[Dict]) -> None:
    """Provides a `st.selectbox` for choosing the model.  Then, if a
    computation graph is available for the selected model, displays the graph
    using `graphviz`.

    Parameters
    ----------
    logs: Iterable[dict]
        Iterable of training logs. Each is a dictionary of training information
    """
    log_names = list(log["name"] for log in logs)
    log_name = st.selectbox("Model:", log_names)
    for log in logs:
        if log["name"] == log_name:
            if "graph" in log.keys():
                st.write(log["graph"])
            else:
                """
                ### Found no graph for this model
                """
            break
