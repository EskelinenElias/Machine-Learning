import numpy as np
from numpy.typing import NDArray as ndarr
from ..activation import Activation

class Sign(Activation):

    def forward(self, inputs: ndarr) -> ndarr:
        return np.sign(inputs)

    def backward(self, loss_gradient: ndarr) -> ndarr:
        return np.zeros_like(loss_gradient)

    def __repr__(self) -> str:
        return "sign"

# eof
