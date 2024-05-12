import os, sys

# import necessary libraries, classes and functions
from numpy import random as rnd
from neural_networks.models import NeuralNetwork
from neural_networks.layers import Dense
from neural_networks.activations import ReLU

# initialize model
model = NeuralNetwork([
    Dense(rnd.randn(1, 3), rnd.randn(3), ReLU()),
    Dense(rnd.randn(3, 3), rnd.randn(3), ReLU()),
    Dense(rnd.randn(3, 1), rnd.randn(1), ReLU()),
])
# test model
inputs = rnd.randn(1, 1)
print(model(inputs))

# eof
