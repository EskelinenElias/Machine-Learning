import numpy as np
from numpy.typing import NDArray as ndarr
from ..activation import Activation

class Softmax(Activation):
    def __init__(self):
        self.probabilities = None

    def forward(self, inputs: ndarr) -> ndarr:
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        self.probabilities = probabilities
        return probabilities

    def backward(self, loss_gradient: ndarr) -> ndarr:
        if self.probabilities is None: raise ValueError("Can't backwards propagate without input.")
        # initialize the output gradient
        softmax_loss_gradient = np.zeros_like(loss_gradient)
        # calculate softmax gradient
        num_classes = np.size(self.probabilities, 1)
        for i in range(np.size(self.probabilities, 0)):
            # calculate the jacobian matrix of softmax with regards to inputs
            probabilities = np.array([self.probabilities[i, :]])
            jacobian_matrix = probabilities.T * (np.eye(num_classes) - probabilities)
            # update the softmax loss gradient using the chain rule
            softmax_loss_gradient[i, :] = loss_gradient[i, :] @ jacobian_matrix
        return softmax_loss_gradient

    def __repr__(self) -> str:
        return 'softmax'

# eof
