import numpy as np
from numpy.typing import NDArray as ndarr
from ..activation import Activation

class Cosine(Activation):
    def __init__(self):
        self.inputs = None

    def forward(self, inputs: ndarr) -> ndarr:
        self.inputs = inputs
        return np.cos(self.inputs)

    def backward(self, loss_gradient: ndarr) -> ndarr:
        if self.inputs is None: raise RuntimeError("Can't backwards propagate without input.")
        inputs_loss_gradient = -np.sin(self.inputs)
        return np.multiply(loss_gradient, inputs_loss_gradient)

    def __repr__(self) -> str:
        return "cosine"

# eof
