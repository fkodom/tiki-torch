from typing import Iterable, Sequence, Dict

import streamlit as st
import plotly.graph_objects as go

from tiki.hut.config import histogram_config


def _write_custom_histogram(logs: Iterable[Dict], tags: Sequence[str], norm="", **config):
    fig = go.Figure()

    for tag in tags:
        split_tag = tag.split(": ")
        if len(split_tag) != 2:
            raise ValueError(
                f"Invalid model name: {split_tag[0]}.  Cannot contain ': '."
            )

        log_name, param_name = split_tag
        for log in logs:
            if log["name"] != log_name:
                continue

            if param_name == "__all__":
                hist_values = []
                for param in log["state_dict"].values():
                    if param is not None:
                        hist_values += param.flatten().tolist()
            else:
                hist_values = log["state_dict"][param_name].flatten().tolist()

            fig.add_trace(
                go.Histogram(
                    x=hist_values,
                    histnorm=norm,
                    name=tag,
                )
            )
        fig.update_layout(
            **config,
            title=norm,
            xaxis=go.layout.XAxis(title="parameter value"),
            yaxis=go.layout.YAxis(title=norm),
        )

    st.write(fig)


def write_histogram(logs: Iterable[Dict]):
    for log in logs:
        if "state_dict" not in log.keys():
            log["state_dict"] = {}
        else:
            log["state_dict"]["__all__"] = None

    params = [f"{log['name']}: {param}" for log in logs for param in log["state_dict"].keys()]
    params = sorted(list(set(params)))
    normalization = st.radio("Normalization", ("probability", "counts", "density"))
    showlegend = st.checkbox("Show legend", value=True)
    histogram_key = 0

    st.write(
        """
        ### Select Parameters
        """
    )
    tags = []
    while len(tags) == 0 or tags[-1] != "-- Select --":
        tags.append(st.selectbox("Parameter:", ["-- Select --"] + params, key=histogram_key))
        histogram_key += 1

    histogram_config["showlegend"] = showlegend
    norm = "" if normalization == "counts" else normalization
    _write_custom_histogram(logs, tags[:-1], norm=norm, **histogram_config)
