"""
original.py
--------------------
Defines DARCNET models for simultaneous PIR denoising & detection.
"""

import torch
import torch.nn as nn

from darcnet.src.network import Network


class ResLayer(nn.Module):
    r"""Defines a single residual layer within DARCNET."""

    def __init__(self):
        super(ResLayer, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.Conv3d(8, 4, 1, padding=0),
            nn.Conv3d(4, 8, 3, padding=1),
            nn.Conv3d(8, 1, 1, padding=0)
        )

    def forward(self, x):
        r"""Pass input datacube `x` through the Residual Layer."""
        return nn.LeakyReLU(0.1)(x + self.conv_layers(x))


class Darcnet(Network):

    def __init__(self):
        super(Darcnet, self).__init__()
        # Build a network of 6 identical residual layers
        self.res_layers = nn.Sequential(
            ResLayer(),
            ResLayer(),
            ResLayer(),
            ResLayer(),
            ResLayer(),
            ResLayer(),
        )

    def get_device_type(self):
        r"""Get data type for the network (i.e. CPU or CUDA)."""
        return self.res_layers[0].conv_layers[0].weight.device.type

    def _forward(self, x):
        """Pass input datacube `x` through DARCNET.  Keep input size to about (100, 100, 100) for training,
        due to large memory footprints.

        :param x: Input values
        :return: Network outputs
        """
        # Push to GPU if necessary
        if self.get_device_type() == 'cuda':
            x = x.cuda()

        # Divide by the temporal standard deviation
        x = x / torch.add(x.std(dim=2, keepdim=True), 1e-4)
        # Forward pass through the residual layers
        x = self.res_layers(x)

        return x

    def forward(self, x):
        """Pass input datacube `x` through DARCNET.  Keep input size to about (100, 100, 100) for training,
        due to large memory footprints.

        :param x: Input values
        :return: Network outputs
        """
        # Push to GPU if necessary
        if self.get_device_type() == 'cuda':
            x = x.cuda()

        # Divide by the temporal standard deviation
        x = x / torch.add(x.std(dim=2, keepdim=True), 1e-4)
        # Forward pass through the residual layers
        x = self.res_layers(x)

        return x

    def get_loss(self, labels, inputs):
        """Modified mean squared error between output and labels inputs.

        :param labels: Ground labels labels
        :param inputs: Training or validation inputs
        :return: Loss tensor
        """
        # Push to GPU if necessary
        if self.get_device_type() == 'cuda':
            labels, inputs = labels.cuda(), inputs.cuda()

        outputs = self._forward(inputs)

        mask = labels > 0.25
        loss = nn.MSELoss()(outputs, labels)
        loss += 0.01 * nn.MSELoss()(outputs[mask], labels[mask])

        return loss
