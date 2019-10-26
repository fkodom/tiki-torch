from abc import ABC


class Callback(ABC):
    """Base class for all callback functions.  Callbacks are functions that are
    automatically executed during network training.

    Execution Times
    ---------------
    Callback execution time is determined from the defined class methods.
    Each callback can define any number of execution methods, which perform
    independent functions during model training.  Method names and their
    corresponding execution times are listed below.

    * **on_start**: Before any training begins
    * **on_end**: Last executed before training terminates
    * **on_epoch**: After each complete epoch of training
    * **on_batch**: After each complete batch during training epochs
    * **on_forward**: After computing outputs for each batch, but before updating parameters

    Attributes
    ----------
    verbose: bool (optional)
        If True, prints a message to the console when an action is performed.
        Nothing is printed if the callback does nothing.  Default: True
    """

    def __init__(self, verbose: bool = True):
        super().__init__()
        self.execution_times = []
        self.verbose = verbose

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def on_start(self, **kwargs):
        pass

    def on_end(self, **kwargs):
        pass

    def on_epoch(self, **kwargs):
        pass

    def on_batch(self, **kwargs):
        pass

    def on_forward(self, **kwargs):
        pass
