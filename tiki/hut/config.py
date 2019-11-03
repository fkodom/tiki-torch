"""
config.py
---------
Default configurations for various figures and widgets in Tiki-Hut.
"""

import plotly.graph_objects as go


__all__ = ["figure_config", "histogram_config"]


# Used only for go.Scatter plots
figure_config = {
    "autosize": False,
    "margin": {"l": 30, "r": 30, "t": 40, "b": 30},
    "legend": go.layout.Legend(x=0.73, y=0.95),
}

# Used only for go.Histogram plots
histogram_config = {
    "autosize": False,
    "margin": {"l": 0, "r": 0, "t": 30, "b": 0},
    "legend": go.layout.Legend(x=0.65, y=0.95),
}
