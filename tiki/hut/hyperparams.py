"""
hyperparams.py
--------------
Method(s) for writing a DataFrame of training hyperparameters to Tiki-Hut
"""

from typing import List, Dict
from collections import OrderedDict

import streamlit as st
from pandas import DataFrame


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = ["write_hyperparams"]


def write_hyperparams(logs: List[Dict]) -> None:
    """Writes a `DataFrame` containing all of the recorded hyperparameters from
    each training log.

    Parameters
    ----------
    logs: Iterable[dict]
        Iterable of training logs. Each is a dictionary of training information
    """
    for log in logs:
        if "hyperparams" not in log.keys():
            log["hyperparams"] = {}

    headers = []
    for log in logs:
        headers += list(log["hyperparams"].keys())

    headers = list(set(headers))
    columns = OrderedDict()
    columns["name"] = [log["name"] for log in logs]

    for log in logs:
        for header in sorted(headers):
            if header not in columns.keys():
                columns[header] = []
            if header in log["hyperparams"].keys():
                columns[header].append(log["hyperparams"][header])
            else:
                columns[header].append(None)

    st.write(DataFrame(columns))
