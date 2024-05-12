import numpy as np
from numpy import ndarray as ndarr
from .layer import Layer
from ..activations import Activation, Identity

class Dense(Layer):
    def __init__(self, weights: ndarr, biases: ndarr, activation: Activation|None = None):
        if len(weights.shape) != 2:
            raise ValueError("Invalid shape of weights matrix.")
        if len(biases.shape) not in (1, 2) or biases.shape[-1] != weights.shape[-1]:
            raise ValueError("Invalid shape of biases vector.")
        # set layer weights and biases
        self._weights = weights
        self._biases = biases.reshape(1, biases.shape[-1])
        # set layer activation 
        self._activation = activation or Identity()
        # prepare variable for saving previous inputs    
        self._inputs = np.zeros((1, weights.shape[0]))


    def forward(self, inputs: ndarr) -> ndarr:
        self._inputs = inputs
        return self.activation(inputs @ self.weights + self.biases)

    def backward(self, output_gradient: ndarr) -> ndarr:
        # calculate loss gradients for inputs, weights and biases
        activation_gradient = self.activation.backward(output_gradient)
        batch_size = np.size(output_gradient, 0)
        weights_gradient = self._inputs.T @ output_gradient / batch_size
        biases_gradient = np.sum(output_gradient, 0, keepdims=True) / batch_size
        inputs_gradient = output_gradient @ self.weights.T
        # update weights and biases and return the inputs gradient
        self.weights -= weights_gradient
        self.biases -= biases_gradient
        return inputs_gradient

    @property
    def weights(self) -> ndarr:
        return self._weights

    @weights.setter
    def weights(self, weights: ndarr):
        if weights.shape == self._weights.shape:
            self._weights = weights
        raise ValueError("Invalid shape")

    @property
    def biases(self) -> ndarr:
        return self._biases

    @biases.setter
    def biases(self, biases: ndarr):
        if biases.shape == self._biases.shape:
            self._biases = biases
        elif len(biases.shape) == 1 and biases.shape[-1] == self._biases.shape[-1]:
            self._biases = biases.reshape(1, biases.shape[-1])
        raise ValueError("Invalid biases vector shape.")

    @property
    def num_inputs(self) -> int:
        return self._weights.shape[0]

    @num_inputs.setter
    def num_inputs(self, num_inputs: int):
        raise AttributeError("Property num_inputs is write-only.")

    @property
    def num_units(self) -> int:
        return self._weights.shape[1]

    @num_units.setter
    def num_units(self, num_units: int):
        raise AttributeError("Property num_units is write-only.")

    @property
    def num_outputs(self) -> int:
        return self._weights.shape[1]

    @num_outputs.setter
    def num_outputs(self, num_outputs: int):
        raise AttributeError("Property num_outputs is write-only.")

    @property
    def activation(self) -> Activation:
        return self._activation

    @activation.setter
    def activation(self, activation: Activation):
        if type(activation) == Activation:
            self._activation = activation
        raise ValueError("Invalid activation function.")

# eof
