"""A single neuron built on Value (Mission 2, micrograd)."""

import random

from value import Value


class Neuron:
    """One weight per input plus a bias; the output is tanh(w1*x1 + ... + wn*xn + b)."""

    def __init__(self, num_inputs: int):
        """Create one random weight per input in w and one random bias b, all Values with data between -1 and 1."""
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.b = Value(random.uniform(-1, 1))
