import plotly.graph_objects as go


__all__ = ["figure_config"]


figure_config = {
    "autosize": False,
    # "width": 350,
    # "height": 275,
    "margin": {"l": 30, "r": 30, "t": 40, "b": 30},
    "legend": go.layout.Legend(x=0.73, y=0.95),
}
