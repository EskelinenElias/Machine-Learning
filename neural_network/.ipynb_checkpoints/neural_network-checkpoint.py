import numpy as np
from numpy import ndarray as ndarr
from .layers import Layer

class NeuralNetwork:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, inputs: ndarr) -> ndarr:
        layer_output = inputs
        for layer in self.layers[:-1]:
            layer_output = layer.forward(layer_output)
        return self.layers[-1].forward(layer_output)

    def __call__(self, inputs: ndarr) -> ndarr:
        return self.forward(inputs)

# eof
