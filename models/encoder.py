# https://github.com/layer6ai-labs/dgm-eval/blob/master/dgm_eval/models/encoder.py

from abc import ABC, abstractmethod

import torch.nn as nn


class Encoder(ABC, nn.Module):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        self.setup(*args, **kwargs)
        self.name = "encoder"

    @abstractmethod
    def setup(self, *args, **kwargs):
        pass

    @abstractmethod
    def transform(self, img):
        """Converts a PIL Image to an input for the model"""
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
