import numpy as np
from numpy.typing import NDArray as  ndarr

class Layer:
    def __init__(self):
        self._num_inputs: int
        self._num_units: int

    def forward(self, inputs: ndarr) -> ndarr:
        pass

    def __call__(self, inputs: ndarr) -> ndarr:
        return self.forward(inputs)

    @property
    def num_inputs(self) -> int:
        return self._num_inputs

    @num_inputs.setter
    def num_inputs(self, num_inputs: int) -> int:
        raise AttributeError("Property num_inputs is write-only.")

    @property
    def num_units(self) -> int:
        return self._num_units

    @num_units.setter
    def num_units(self, num_units: int) -> int:
        raise AttributeError("Property num_units is write-only.")

    @property
    def num_outputs(self) -> int:
        return self._num_units

    @num_outputs.setter
    def num_outputs(self, num_outputs: int) -> int:
        raise AttributeError("Property num_outputs is write-only.")

# eof
