import numpy as np
from numpy.typing import NDArray as ndarr
from ..activation import Activation

class Identity(Activation):
    def forward(self, inputs: ndarr) -> ndarr:
        return inputs

    def backward(self, loss_gradient: ndarr) -> ndarr:
        return loss_gradient

    def __repr__(self) -> str:
        return "identity"
