import numpy as np
from numpy.typing import NDArray as ndarr
from ..activation import Activation

class Swish(Activation):
    def __init__(self):
        self.inputs = None
        self.sigmoid = None

    def forward(self, inputs: ndarr) -> ndarr:
        self.inputs = inputs
        self.sigmoid = 1 / (1 + np.exp(-inputs))
        return self.inputs * self.sigmoid

    def backward(self, loss_gradient: ndarr) -> ndarr:
        if self.inputs is None or self.sigmoid is None:
            raise RuntimeError("Can't backwards propagate without input.")
        inputs_gradient = self.sigmoid + self.inputs * self.sigmoid * (1 - self.sigmoid)
        return np.multiply(loss_gradient, inputs_gradient)

    def __repr__(self) -> str:
        return 'swish'
