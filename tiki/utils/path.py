from datetime import datetime


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = ["custom_path"]


def custom_path(path: str, **kwargs) -> str:
    """TODO: Documentation"""
    if "{codename}" in path and "codename" in kwargs.keys():
        path = path.replace("{codename}", kwargs["codename"])
    if "{epoch}" in path and "epoch" in kwargs.keys():
        path = path.replace("{epoch}", kwargs["epoch"])
    if "{datetime}" in path:
        datetime_str = str(datetime.now())[:19]
        datetime_str = datetime_str.replace(" ", "-").replace(":", "-")
        path = path.replace("{datetime}", datetime_str)

    return path
