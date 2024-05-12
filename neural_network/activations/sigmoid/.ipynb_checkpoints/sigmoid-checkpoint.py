import numpy as np
from numpy import ndarray
from ..activation_function import ActivationFunction

class Sigmoid(ActivationFunction):
    def __init__(self):
        self.inputs = None

    def forward(self, inputs: ndarray) -> ndarray:
        self.inputs = inputs
        return 1 / (1 + np.exp(-inputs))

    def backward(self, loss_gradient: ndarray) -> ndarray:
        sigmoid_at_inputs = self.forward(self.inputs)
        inputs_gradient = np.multiply(sigmoid_at_inputs, (1 - sigmoid_at_inputs))
        return np.multiply(loss_gradient, inputs_gradient)

    def __repr__(self) -> str:
        return 'sigmoid'
