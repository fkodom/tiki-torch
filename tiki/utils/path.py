from datetime import datetime


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = ["custom_path"]


def custom_path(path: str, **kwargs) -> str:
    """Customizes a string by replacing keywords with specified values.
    Typically used for creating unique save paths for training logs, model
    checkpoints, etc.  Allowed keywords: ["{codename}", "{epoch}", "{date}",
    "{time}", "{datetime}"]

    Parameters
    ----------
    path: str
        String to be customized
    kwargs
        Keyword arguments used for customization

    Examples
    --------
    >>> custom_path("{codename}.txt", codename="narwhal")
    'narwhal.txt'
    >>> custom_path("epoch-{epoch}.txt", epoch=4)
    'epoch-4.txt'
    """
    if "{codename}" in path and "codename" in kwargs.keys():
        path = path.replace("{codename}", str(kwargs["codename"]))
    if "{epoch}" in path and "epoch" in kwargs.keys():
        path = path.replace("{epoch}", str(kwargs["epoch"]))
    if "{date}" in path:
        date_str = datetime.now().strftime('%d %b %y')
        path = path.replace("{date}", date_str)
    if "{time}" in path:
        time_str = datetime.now().strftime('%H-%M-%S')
        path = path.replace("{time}", time_str)
    if "{datetime}" in path:
        datetime_str = datetime.now().strftime('%d %b %y %H-%M-%S')
        path = path.replace("{datetime}", datetime_str)

    return path
