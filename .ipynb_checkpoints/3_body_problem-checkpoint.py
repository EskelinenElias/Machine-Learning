import numpy as np
from numpy.typing import NDArray as ndarr
from scipy.stats import levy_stable as stable
from time import time

# import neural network
from neural_network.models import NeuralNetwork
from neural_network.layers import Layer, Dense
from neural_network.activations import Activation, ReLU, Tanh, Sign

# define function for generating symmetric stable samples
def sym_stable_fast(stability: float, scale: float, shape: tuple) -> ndarr:
    if stability <= 0 or stability > 2:
        raise ValueError("stability must be greater than 0 and less than or equal to 2.")
    if scale <= 0:
        raise ValueError("scale must be greater than 0.")
    sample_size = np.prod(shape)
    step_size = int(5*np.log(sample_size)*np.sqrt(sample_size))
    sample = np.zeros(sample_size)
    i = 0
    while (sample_size - i >= step_size):
        sample[i:i+step_size] = stable.rvs(stability, 0, 0, scale, size=step_size)
        i += step_size
    if i < sample_size:
        sample[i:sample_size] = stable.rvs(stability, 0, 0, scale, size=sample_size-i)
    return sample.reshape(shape)

def sym_stable_slow(stability: float, scale: float, shape: tuple) -> ndarr:
    if stability <= 0 or stability > 2:
        raise ValueError("stability must be greater than 0 and less than or equal to 2.")
    if scale <= 0:
        raise ValueError("scale must be greater than 0.")
    if (sample:=stable.rvs(stability, 0, 0, scale, size=shape)) is None:
        raise ValueError(f"Invalid shape: {shape}")
    return sample

# initialize model


print(1e3/np.log(1e3), 1e8/np.log(1e8), 1e10/np.log(1e10))


start_time = time()
sym_stable_fast(1.5, 1, (10000, 10000))
print(f"Time: {time()-start_time:.2f} s")

start_time = time()
#sym_stable_slow(1.5, 1, (5000, 10000))
print(f"Time: {(time()-start_time)*2:.2f} s (simulated)")
# eof
