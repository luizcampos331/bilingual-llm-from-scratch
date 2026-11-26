"""A layer of neurons built on Neuron (Mission 2, micrograd)."""

from neuron import Neuron


class Layer:
    """A list of neurons that all receive the same inputs; the output is one Value per neuron."""

    def __init__(self, nin: int, nout: int):
        """Create nout neurons, each one a Neuron with nin weights (one per input)."""
        self.neurons = [Neuron(nin) for _ in range(nout)]
