"""
histogram.py
------------
Methods for writing histograms of trainable parameters to Tiki-Hut
"""

from typing import Iterable, Sequence, Dict

import streamlit as st
import plotly.graph_objects as go

from tiki.hut.config import histogram_config


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = ["write_histogram"]


def _write_custom_histogram(
    logs: Iterable[Dict], tags: Sequence[str], norm: str = "", **config
) -> None:
    """Writes a histogram for visualizing one or more trainable parameters.
    The parameter names are provided in the `tags` argument, which is
    automatically collected in the `write_histogram` method (below).

    Parameters
    ----------
    logs: Iterable[dict]
        Iterable of training logs. Each is a dictionary of training information
    tags: Sequence[str]
        Sequence of parameter names to visualize in the histogram
    norm: str, optional
        String specifying the normalization method for the histogram.  Available
        options: ["", "probability", "density"].  Default: "" (counts).
    **config
        Additional keyword arguments for customizing the histogram figure.
    """
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

            fig.add_trace(go.Histogram(x=hist_values, histnorm=norm, name=tag))
        fig.update_layout(
            **config,
            title=norm,
            xaxis=go.layout.XAxis(title="parameter value"),
            yaxis=go.layout.YAxis(title=norm),
        )

    st.write(fig)


def write_histogram(logs: Iterable[Dict]) -> None:
    """Provides one or more `st.selectbox` items for choosing trainable
    parameters to visualize using a histogram.  Also provides user options for
    the histogram normalization and legend visibility.

    Parameters
    ----------
    logs: Iterable[dict]
        Iterable of training logs. Each is a dictionary of training information
    """
    for log in logs:
        if "state_dict" not in log.keys():
            log["state_dict"] = {}
        else:
            log["state_dict"]["__all__"] = None

    params = [
        f"{log['name']}: {param}" for log in logs for param in log["state_dict"].keys()
    ]
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
    default_tag = "-- Select --"
    while len(tags) == 0 or tags[-1] != default_tag:
        tags.append(
            st.selectbox("Parameter:", [default_tag] + params, key=histogram_key)
        )
        histogram_key += 1

    histogram_config["showlegend"] = showlegend
    norm = "" if normalization == "counts" else normalization
    _write_custom_histogram(logs, tags[:-1], norm=norm, **histogram_config)
