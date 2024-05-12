import numpy as np
from numpy.typing import NDArray as ndarr
from ..layers import Layer

class NeuralNetwork:
    def __init__(self, layers: list[Layer]):
        self.layers = layers # store all network layers
        self.inputs = None # store inputs to input layer

    def forward(self, inputs: ndarr) -> ndarr:
        # store inputs for later use and
        self.inputs = inputs
        # prepare a temporary variable for calculation
        hidden_layer_output = self.inputs
        # calculate hidden layer output
        for layer in self.hidden_layers:
            hidden_layer_output = layer(hidden_layer_output)
        # return output layer output
        return self.output_layer(hidden_layer_output)

    # default method of the class; class instance is callable
    def __call__(self, inputs: ndarr) -> ndarr:
        return self.forward(inputs)

    # property hidden_layers returns the hidden layers of the network
    @property
    def hidden_layers(self) -> list[Layer]:
        return self.layers[:-1]

    # property output_layer returns the output layer of the network
    @property
    def output_layer(self) -> Layer:
        return self.layers[-1]

    @property
    def input_size(self) -> int:
        return self.layers[0].num_inputs

    @property
    def output_size(self) -> int:
        return self.layers[-1].num_outputs

# eof
