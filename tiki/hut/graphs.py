from typing import Iterable, Dict

import streamlit as st


def write_graphs(logs: Iterable[Dict]):
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
