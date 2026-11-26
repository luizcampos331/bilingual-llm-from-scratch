"""A multi-layer perceptron built on Layer (Mission 2, micrograd)."""

from collections.abc import Sequence

from layer import Layer
from value import Value


class MLP:
    """A stack of layers where the outputs of one layer are the inputs of the next."""

    def __init__(self, nin: int, nouts: list[int]):
        """Create one Layer per size in nouts, each taking as inputs the number of neurons of the layer before it (nin for the first)."""
        layer_sizes = [nin] + nouts
        self.layers = [
            Layer(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(nouts))
        ]

    def __call__(self, x: Sequence[float | Value]) -> Sequence[float | Value]:
        for layer in self.layers:
            x = layer(x)
        return x
