import numpy as np
from numpy.typing import NDArray as ndarr
from ..activation import Activation

class Step(Activation):

    def forward(self, inputs: ndarr) -> ndarr:
        return inputs > 0

    def backward(self, loss_gradient: ndarr) -> ndarr:
        return np.zeros_like(loss_gradient)

    def __repr__(self) -> str:
        return "step"

# eof
