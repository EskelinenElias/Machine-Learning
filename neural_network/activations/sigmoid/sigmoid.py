import numpy as np
from numpy.typing import NDArray as ndarr
from ..activation import Activation

class Sigmoid(Activation):
    def __init__(self):
        self.inputs = None

    def forward(self, inputs: ndarr) -> ndarr:
        self.inputs = inputs
        return 1 / (1 + np.exp(-inputs))

    def backward(self, loss_gradient: ndarr) -> ndarr:
        if self.inputs is None: raise RuntimeError("Can't backpropagate without input.")
        sigmoid_at_inputs = self.forward(self.inputs)
        inputs_gradient = np.multiply(sigmoid_at_inputs, (1 - sigmoid_at_inputs))
        return np.multiply(loss_gradient, inputs_gradient)

    def __repr__(self) -> str:
        return 'sigmoid'
