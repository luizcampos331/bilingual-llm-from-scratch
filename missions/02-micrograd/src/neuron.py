"""A single neuron built on Value (Mission 2, micrograd)."""

import random
from collections.abc import Sequence

from value import Value


class Neuron:
    """One weight per input plus a bias; the output is tanh(w1*x1 + ... + wn*xn + b)."""

    def __init__(self, num_inputs: int):
        """Create one random weight per input in w and one random bias b, all Values with data between -1 and 1."""
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: Sequence[float | Value]) -> Value:
        """Multiply each input by its weight, add them all up with the bias b, and return the tanh of that sum as a Value."""
        return sum((wi * xi for wi, xi in zip(self.w, x)), self.b).tanh()
