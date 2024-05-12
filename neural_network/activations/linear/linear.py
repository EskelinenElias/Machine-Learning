import numpy as np
from numpy.typing import NDArray as ndarr
from ..activation import Activation

class Linear(Activation):
    def __init__(self, a: float = 1, b: float = 0):
        self.a = a
        self.b = b

    def forward(self, inputs: ndarr) -> ndarr:
        return self.a * inputs + self.b

    def backward(self, loss_gradient: ndarr) -> ndarr:
        return self.a * loss_gradient

    def __repr__(self) -> str:
        return "linear"
