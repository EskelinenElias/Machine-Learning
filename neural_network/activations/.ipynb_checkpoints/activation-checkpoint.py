from typing import Callable
from numpy.typing import NDArray as ndarr

class Activation:
    def __init__(self):
        pass

    def forward(self, inputs: ndarr) -> ndarr:
        return self.activation(inputs)

    def backward(self, loss_gradient: ndarr) -> ndarr:
        return self.gradient(loss_gradient)

    def __call__(self, inputs: ndarr) -> ndarr:
        return self.forward(inputs)

    def __repr__(self) -> str:
        return "activation function"

    # eof
