"""
trainers.py
-------
Base trainer module for all models in `tiki`.
"""

from typing import Iterable, Sequence, Callable, List

from tqdm import tqdm
from torch import nn, optim, Tensor
from torch.utils.data import Dataset

from tiki.callbacks import Callback
from tiki.base.base import BaseTrainTest
from tiki.base.utils import setup
from tiki.utils.data import get_data_loaders


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = ["Trainer"]


# Define batch datatype (used for internal methods).
# Each batch is an iterable (over train, validation sets) of Tensors.
# If the inputs have inconsistent sizes, lists of Tensors are used instead.
Batch = Sequence[Tensor or List[Tensor]]


class Trainer(BaseTrainTest):
    """Basic neural network trainer for supervised and unsupervised applications.
    Supports a wide variety of neural network types, including fully-connected,
    CNN, and RNN.
    """

    def __init__(self):
        super().__init__()

    def epoch(
        self,
        model: nn.Module,
        tr_dataset: Dataset = (),
        va_dataset: Dataset = (),
        gpus: int or Sequence[int] = (),
        batch_size: int = 20,
        shuffle: bool = True,
        progress_bar: bool = True,
        verbosity: int = 1,
    ) -> bool:
        """Performs a full epoch of training and validation.

        Parameters
        ----------
        model: nn.Module
            PyTorch module to train
        tr_dataset: Dataset
            Dataset of inputs (one or more) and labels for training.  Labels
            should be provided as the *last* argument to the Dataset.
        va_dataset: Dataset
            Dataset of inputs (one or more) and labels for validation.  Labels
            should be provided as the *last* argument to the Dataset.
        gpus: int or Sequence[int]
            If an `int` is provided, specifies the number of GPUs to use
            during training.  GPUs are chosen in order of ascending device ID.
            If a sequence of ints is provided, specifies the exact device IDs of
            the devices to use during training.
        batch_size: int, optional
            Batch size for training.  Default: 25
        shuffle: bool, optional
            If True, automatically shuffles training and validation examples
            at the start of each epoch.  Default: True
        progress_bar: bool
            If True, a progress bar with performance metrics is displayed
            during each epoch.
        verbosity: int
            If progress_bar == False, determines the frequency at which
            performance metrics are printed.  Ignored if progress_bar == True.

        Returns
        -------
        bool
            Indicates whether training should be terminated using 'break'
            statements.  Typically only happens on early stopping or Exceptions.

        Raises
        ------
        ValueError
            If keyword argument 'loss' is not provided
        NotImplementedError
            If the 'forward' method has not been implemented for sub-classes
        """
        self.info["epochs"] += 1
        epoch = self.info["epochs"]

        tr_loader, va_loader = get_data_loaders(
            (tr_dataset, va_dataset),
            batch_size=batch_size,
            shuffle=shuffle,
        )
        progress_bar = tqdm(
            total=min(len(tr_loader), len(va_loader)),
            desc=f"Ep {epoch}",
            dynamic_ncols=True,
            disable=not progress_bar
        )

        postfix = ""
        for train, val in zip(tr_loader, va_loader):
            break_flag = self._run_on_batch(model, train=train, val=val, gpus=gpus)
            if break_flag:
                progress_bar.close()
                return True

            postfixes = [f"{k}: {v:+.3e}" for k, v in self.metrics.items()]
            postfix = ", ".join(postfixes)
            progress_bar.update()
            progress_bar.set_postfix_str(postfix)

        self._update_history()
        progress_bar.close()
        if not progress_bar and epoch % verbosity == 0:
            print(f"Ep {epoch} : {postfix}")

        return self._execute_callbacks(model, execution_times=["on_epoch"])

    # TODO: Fix type annotation for `loss`
    # (Dependent on MyPy type checking)
    def train(
        self,
        model: nn.Module,
        tr_dataset: Dataset = (),
        va_dataset: Dataset = (),
        loss: object = None,
        optimizer: str or optim.Optimizer = "adam",
        gpus: int or Sequence[int] = 0,
        epochs: int = 10,
        batch_size: int = 25,
        shuffle: bool = True,
        alpha: float = 0.98,
        metrics: Iterable[str or Callable] = (),
        callbacks: Iterable[str or Callback] = (),
        progress_bar: bool = True,
        verbosity: int = 1,
    ) -> None:
        """Initiates a complete training sequence for the network.

        Parameters
        ----------
        model: nn.Module
            PyTorch module to train
        tr_dataset: Dataset
            Dataset of inputs (one or more) and labels for training.  Labels
            should be provided as the *last* argument to the Dataset.
        va_dataset: Dataset, optional
            Dataset of inputs (one or more) and labels for validation.  Labels
            should be provided as the *last* argument to the Dataset.
        loss: Callable
            Loss function used for computing training and validation error.
            **If not specified, this function will raise an exception.**
        optimizer: str or optim.Optimizer, optional
            Optimization algorithm to use for network training.  Can be provided
            either as an 'optim.Optimizer' instance, or a string specifier. If
            not specified, defaults to 'optim.Adam' with its default arguments.
        gpus: int or Sequence[int]
            If an `int` is provided, specifies the number of GPUs to use
            during training.  GPUs are chosen in order of ascending device ID.
            If a sequence of ints is provided, specifies the exact device IDs of
            the devices to use during training.
        epochs: int, optional
            Number of epochs for training.  Default: 10
        batch_size: int, optional
            Batch size for training.  Default: 25
        shuffle: bool, optional
            If True, automatically shuffles training and validation examples
            at the start of each epoch.  Default: True
        alpha: float, optional
            Controls how quickly loss values are updated using an IIR filter.
            Range: [0, 1].  Close to 1 gives fast update, but low dependence on
            previous batches.  Close to 0 gives slow update, but incorporates
            information from many previous batches.  Default: 0.95
        metrics: Iterable[str or Callable], optional
            Iterable of performance metrics to compute for each batch of
            validation data.  If strings are provided, will attempt to retrieve
            the corresponding metric function from tiki.metrics.
        callbacks: Iterable[Callable], optional
            Iterable of callable functions to automatically execute during training
        progress_bar: bool
            If True, a progress bar with performance metrics is displayed
            during each epoch.
        verbosity: int
            If progress_bar == False, determines the frequency at which
            performance metrics are printed.  Ignored if progress_bar == True.

        Raises
        ------
        ValueError
            If keyword argument 'loss' is not provided
        NotImplementedError
            If the 'forward' method has not been implemented for sub-classes
        """
        self.alpha = alpha
        dp_model, self.loss, self.optimizer, self.callbacks, self.metric_fns = setup(
            model=model,
            loss=loss,
            optimizer=optimizer,
            callbacks=callbacks,
            metrics=metrics,
            gpus=gpus,
        )
        self._execute_callbacks(model, execution_times=["on_start"])

        for _ in range(epochs):
            break_flag = self.epoch(
                dp_model,
                tr_dataset=tr_dataset,
                va_dataset=va_dataset,
                gpus=gpus,
                batch_size=batch_size,
                shuffle=shuffle,
                progress_bar=progress_bar,
                verbosity=verbosity,
            )
            if break_flag:
                break

        self._execute_callbacks(model, callbacks=callbacks, execution_times=["on_end"])
