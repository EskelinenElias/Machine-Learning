import numpy as np
from numpy.typing import NDArray as ndarr
from ..activation import Activation

class LeakyReLU(Activation):
    def __init__(self, leakiness: float):
        self.leakiness = leakiness
        self.inputs = np.empty(0)

    def forward(self, inputs: ndarr) -> ndarr:
        self.inputs = inputs
        return np.maximum(self.leakiness*inputs, inputs)

    def backward(self, loss_gradient: ndarr) -> ndarr:
        inputs_gradient = np.where(self.inputs > 0, 1, self.leakiness)
        return np.multiply(loss_gradient, inputs_gradient)

    def __repr__(self) -> str:
        return 'leaky ReLU'

# eof
