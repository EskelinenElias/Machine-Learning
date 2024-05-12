import numpy as np
from numpy.typing import NDArray as ndarr
from ..activation import Activation

class ReLU(Activation):
    def __init__(self):
        self.inputs = None

    def forward(self, inputs: ndarr) -> ndarr:
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, loss_gradient: ndarr) -> ndarr:
        if self.inputs is None: raise RuntimeError("Cannot backward propagate without input.")
        inputs_gradient: ndarr = np.where(self.inputs > 0, 1, 0)
        return np.multiply(loss_gradient, inputs_gradient)

    def __repr__(self) -> str:
        return 'ReLU'
