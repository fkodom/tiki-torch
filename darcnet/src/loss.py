import torch
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, labels):
        return -torch.mean(labels * (1 - inputs).pow(self.gamma) * torch.log(inputs) +
                           (1 - labels) * inputs.pow(self.gamma) * torch.log(1 - inputs))
