"""A layer of neurons built on Neuron (Mission 2, micrograd)."""

from collections.abc import Sequence

from neuron import Neuron
from value import Value


class Layer:
    """A list of neurons that all receive the same inputs; the output is one Value per neuron."""

    def __init__(self, nin: int, nout: int):
        """Create nout neurons, each one a Neuron with nin weights (one per input)."""
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: Sequence[float | Value]) -> list[Value]:
        """Give the same inputs x to every neuron and return their outputs as a list of Values, one per neuron."""
        return [neuron(x) for neuron in self.neurons]
