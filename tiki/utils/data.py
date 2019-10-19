"""
data.py
-------
Defines a set of helper functions/classes for efficiently sampling data during
model training.
"""

from typing import Sequence
from math import ceil
from functools import partial

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import List


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = ["ListDataset", "ListDataLoader", "get_data_loaders"]


class ListDataset(Dataset):
    """Dataset containing lists of Tensors, which is particularly useful when
    example inputs/labels have inconsistent sizes.  The conjunction of
    ListDataset/ListDataLoader works as a drop-in replacement for
    Dataset/DataLoader, which is standard for regularly sized data.

    Parameters
    ----------
    tensor_lists: Iterable[List[Tensor]]
        An iterable containing lists of Tensors.  Lists should have equal
        length, so they can be sampled simultaneously during model training.
    """

    def __init__(self, *tensor_lists: List[Tensor]):
        super().__init__()
        self.tensor_lists = tensor_lists

    def __len__(self):
        return min(len(tensor_list) for tensor_list in self.tensor_lists)

    def __getitem__(self, item):
        if isinstance(item, slice) or isinstance(item, int):
            return tuple(x[item] for x in self.tensor_lists)
        else:
            return tuple(list(map(x.__getitem__, item)) for x in self.tensor_lists)


class ListDataLoader(object):
    """DataLoader for efficiently sampling data from ListDatasets.  The
    conjunction of ListDataset/ListDataLoader works as a drop-in replacement for
    Dataset/DataLoader, which is standard for regularly sized data.

    Parameters
    ----------
    dataset: ListDataset
        ListDataset that needs to be sampled during training
    batch_size: int
        Number of samples per training batch
    shuffle: bool
        If True, samples are shuffled into random order
    """

    def __init__(
        self, dataset: ListDataset, batch_size: int = 25, shuffle: bool = True
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.batch = 0
        self.num_examples = len(dataset)
        self.num_batches = ceil(self.num_examples / batch_size)

        if shuffle:
            self.batch_idx = torch.randperm(self.num_batches) * batch_size
        else:
            self.batch_idx = torch.arange(self.num_batches) * batch_size

    def _get_batch_indices(self):
        if self.shuffle:
            self.batch_idx = torch.randperm(self.num_batches) * self.batch_size
        else:
            self.batch_idx = torch.arange(self.num_batches) * self.batch_size

    def __len__(self):
        return self.num_examples

    def __iter__(self):
        self.batch = 0
        return self

    def __next__(self):
        if self.batch < self.num_batches:
            max_idx = min(self.batch + self.batch_size, self.num_examples)
            idx = self.batch_idx[self.batch : max_idx]
            self.batch += 1
            return self.dataset[idx]
        else:
            self._get_batch_indices()
            self.batch = 0
            raise StopIteration


def get_data_loaders(
    datasets: Sequence[Dataset],
    batch_size: int = 20,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Sequence[DataLoader]:
    """Prepares DataLoaders for a sequence of Datasets.  The first returned
    DataLoader will have batches of size 'batch_size'. All others will have
    the same number of total batches, which determines their batch sizes.

    Parameters
    ----------
    datasets: Sequence[Dataset]
        A sequence of Datasets that DataLoaders will be created for
    batch_size: int (optional)
        Batch size for training
    shuffle: bool (optional)
        If True, automatically shuffles training and validation examples
        at the start of each epoch.  Default: True
    num_workers: int (optional)
        Number of 'torch.multiprocessing' workers used for data loading.
        Used by the DataLoader object to avoid the GIL.  Default: 2

    Returns
    -------
    DataLoader, DataLoader or Iterable
        Always returns a DataLoader for training data as the first return
        value.  If validation is defined, also returns a DataLoader for
        validation data.  Else, returns an Iterator of (None, None) tuples.
    """
    data_loaders = []
    num_datasets = len(datasets)

    if len(datasets) == 0:
        return []
    elif isinstance(datasets[0], ListDataset):
        get_data_loader = partial(ListDataLoader, shuffle=shuffle)
    elif isinstance(datasets[0], Dataset):
        get_data_loader = partial(DataLoader, shuffle=shuffle, num_workers=num_workers)
    else:
        raise ValueError(f"Dataset type {type(datasets[0])} not supported.")

    data_loaders.append(get_data_loader(datasets[0], batch_size=batch_size))
    num_batches = len(data_loaders[0])

    for i in range(1, num_datasets):
        if not datasets[i]:
            data_loaders.append([(None, ) for _ in range(num_batches)])
        else:
            new_batch_size = max(1, len(datasets[i]) // num_batches)
            data_loaders.append(get_data_loader(datasets[i], batch_size=new_batch_size))

    return tuple(data_loaders)
