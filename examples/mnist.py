"""
tiki.py
-------
Example script using `tiki` to train a MNIST handwritten digits classifier.
"""

import torch.nn as nn
from tiki.trainers import BaseTrainer

# Torchvision is not need to install tiki, but is required for this example.
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST


class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 20),
            nn.Linear(20, 10),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x.view(-1, 784))


if __name__ == "__main__":
    tr_dataset = MNIST(root="data", train=True, transform=ToTensor(), download=True)
    va_dataset = MNIST(root="data", train=False, transform=ToTensor(), download=True)

    net = MnistNet()
    BaseTrainer().train(
        net,
        tr_dataset=tr_dataset,
        va_dataset=va_dataset,
        loss="cross_entropy",
        optimizer="adam",
        gpus=[0],
        metrics=["sparse_cat_acc"],
        callbacks=[
            "terminate_on_nan",
            "early_stopping",
        ]
    )
