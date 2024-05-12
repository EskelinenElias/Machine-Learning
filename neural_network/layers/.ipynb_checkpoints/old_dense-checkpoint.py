import numpy as np
from numpy import ndarray as ndarr
from .layer import Layer
from ..activation_functions import ActivationFunction as AFun, Identity

class Dense(Layer):
    def __init__(self, weights: ndarr, biases: ndarr, \
    activation_function: AFun = Identity()):
        self.parameters = (weights, biases)
        self._num_inputs, self._num_units = weights.shape
        self.activation_function = activation_function

    @property
    def weights(self) -> ndarr:
        return self._weights

    @weights.setter
    def weights(self, weights: ndarr):
        if weights.shape == (self.num_inputs, self.num_units):
            self._weights = weights
        else:
            raise AttributeError("Invalid shape of weights matrix.")

    @property
    def biases(self) -> ndarr:
        return self._biases

    @biases.setter
    def biases(self, biases: ndarr):
        if biases.shape == (1, self._num_units):
            self._biases = biases
        else:
            raise AttributeError("Invalid shape of biases vector.")

    @property
    def parameters(self) -> ndarr:
        return np.concatenate((self.weights, self.biases), 0)

    @parameters.setter
    def parameters(self, parameters: ndarr|tuple):
        if type(parameters) == ndarr:
            if len(parameters.shape) != 2:
                raise ValueError("Invalid shape of parameters matrix.")
            if parameters.shape[1] < 2:
                raise ValueError("Not enough parameters.")
            self._weights = parameters[:-1, :]
            self._biases = parameters[-1, :]
            self._num_inputs, self._num_units = self._weights.shape
        elif type(parameters) == tuple:
            if len(parameters) != 2:
                raise ValueError("Not enough parameters.")
            weights, biases = parameters
            if len(weights.shape) != 2:
                raise ValueError("Invalid shape of weights matrix.")
            if len(biases.shape) != 1 and len(biases.shape) != 2:
                raise ValueError("Invalid shape of biases vector.")
            if weights.shape[-1] != biases.shape[-1]:
                raise ValueError("The shapes of weights matrix and biases vector do not match.")
            self._weights = weights
            if len(biases.shape) == 1:
                self._biases = biases.reshape(1, biases.shape[0])
            else:
                self._biases = biases
            self._num_inputs, self._num_units = self._weights.shape

    def forward(self, inputs: ndarr) -> ndarr:
        weighted_inputs = np.dot(inputs, self.weights) + self.biases
        return self.activation_function.forward(weighted_inputs)
