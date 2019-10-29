import plotly.graph_objects as go


__all__ = ["figure_config", "histogram_config"]


figure_config = {
    "autosize": False,
    "margin": {"l": 30, "r": 30, "t": 40, "b": 30},
    "legend": go.layout.Legend(x=0.73, y=0.95),
}

histogram_config = {
    "autosize": False,
    "margin": {"l": 30, "r": 30, "t": 40, "b": 30},
    "legend": go.layout.Legend(x=0.65, y=0.95),
}
