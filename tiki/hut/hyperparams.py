from typing import List, Dict
from collections import OrderedDict

import streamlit as st
from pandas import DataFrame


def write_hyperparams(logs: List[Dict]):
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
