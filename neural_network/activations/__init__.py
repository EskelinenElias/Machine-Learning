# parent class
from .activation import Activation

# linear activations
from .linear.identity import Identity
from .linear.linear import Linear

# ReLU activations
from .ReLU.ReLU import ReLU
from .ReLU.leaky_ReLU import LeakyReLU

# Sigmoid activations
from .sigmoid.sigmoid import Sigmoid
from .sigmoid.tanh import HyperbolicTangent
from .sigmoid.tanh import HyperbolicTangent as Tanh
from .sigmoid.tanh_normalized import NormalizedHyperbolicTangent

# step function activations
from .step.sign import Sign
from .step.step import Step

# classification activations
from .classification.softmax import Softmax

# other
from .other.swish import Swish

# wave activations
from .wave.sine import Sine
from .wave.cosine import Cosine
