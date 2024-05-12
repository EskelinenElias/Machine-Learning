import numpy as np
from numpy.typing import NDArray as ndarr
from ..activation import Activation

class NormalizedHyperbolicTangent(Activation):
    def __init__(self):
        self.inputs = None

    def forward(self, inputs: ndarr) -> ndarr:
        self.inputs = inputs
        return (np.tanh(inputs) + 1) / 2

    def backward(self, loss_gradient: ndarr) -> ndarr:
        if self.inputs is None: raise RuntimeError("Can't backpropagate without input.")
        inputs_gradient = (1 - np.tanh(self.inputs)**2) / 2
        return np.multiply(loss_gradient, inputs_gradient)

    def __repr__(self) -> str:
        return 'tanh normalized'
