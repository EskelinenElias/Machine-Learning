import numpy as np
from numpy.typing import NDArray as ndarr
from typing import Callable

class SimplePerceptron:
    def __init__(self, weights: ndarr, activation: Callable):
        self.weights = weights
        self.activation = activation

    def forward(self, inputs: ndarr) -> float:
        return self.activation(inputs @ self.weights)

    def __call__(self, inputs: ndarr) -> float:
        return self.forward(inputs)

    @property
    def weights(self) -> ndarr:
        return self._weights

    @weights.setter
    def weights(self, weights: ndarr):
        if len(weights.shape) > 1:
            raise ValueError(f"Invalid shape of weights vector: {weights.shape}.")
        self._weights = weights

class Perceptron:
    def __init__(self, weights: ndarr, bias: float, activation: Callable):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def forward(self, inputs: ndarr) -> float:
        return self.activation(inputs @ self.weights + self.bias)

    def __call__(self, inputs: ndarr) -> float:
        return self.forward(inputs)

if __name__ == "__main__":

    # initialize model
    weights = np.random.randn(3)
    activation = np.tanh
    # bias = np.random.randn()
    model = SimplePerceptron(weights, activation)

    # calculate model output
    inputs = np.random.randn(3)
    print(model(inputs))

    # change weights
    weights = np.random.randn(3)
    model.weights = weights

    # change weights
    weights = np.random.randn(1, 3)
    model.weights = weights

# eof
