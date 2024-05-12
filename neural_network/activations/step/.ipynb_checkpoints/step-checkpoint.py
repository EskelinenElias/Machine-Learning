import numpy as np
from numpy import ndarray as ndarr
from ..activation_function import ActivationFunction

class Step(ActivationFunction):

    def forward(self, inputs: ndarr) -> ndarr:
        return inputs > 0

    def backward(self, loss_gradient: ndarr) -> ndarr:
        return np.zeros_like(loss_gradient)

    def __repr__(self) -> str:
        return "step"

# eof
