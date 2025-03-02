import torch
import torch.nn as nn

class NullBaseline(nn.Module):
    def __init__(self, config):
        super(NullBaseline, self).__init__()

    def forward(self, input):
        return input

