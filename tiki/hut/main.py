import streamlit as st
import torch
from torchviz import make_dot

import plotly.graph_objects as go


"""
# Streamlit Demo

Example for how to use the `streamlit` Python API.
"""

net1 = torch.nn.Linear(10, 5)
net2 = torch.nn.Sequential(
    torch.nn.Linear(10, 5),
    torch.nn.Linear(5, 2),
)
models = {"net1": net1, "net2": net2}

page = st.selectbox("Viewing: ", ("Model Graphs", "Metrics", "Third"))

if page == "Model Graphs":
    model = st.selectbox("Model:", list(models.keys()))
    net = models[model]
    out = net(torch.randn(20, 10)).mean()
    st.write(make_dot(out, params=dict(net.named_parameters())))

if page == "Metrics":
    loss_checkbox = st.checkbox("Show losses:", value=True)
    if loss_checkbox:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(1, 11)),
                y=[1 / (z ** 2) for z in range(1, 11)],
            )
        )
        fig.update_layout(margin={"l": 30, "r": 30, "t": 30, "b": 30})
        st.write(fig)
