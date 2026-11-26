from layer import Layer
from neuron import Neuron


class TestLayer:
    def test_layer_starts_with_one_neuron_per_output(self):
        """Given Layer(2, 3), expect 3 neurons, each a Neuron with 2 weights (one per input)."""
        layer = Layer(2, 3)
        assert len(layer.neurons) == 3
        for neuron in layer.neurons:
            assert len(neuron.w) == 2
            assert isinstance(neuron, Neuron)
