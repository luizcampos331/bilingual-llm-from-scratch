from neuron import Neuron
from value import Value


class TestNeuron:
    def test_neuron_starts_with_one_weight_per_input_and_a_bias(self):
        """Given Neuron(1), expect 1 weight in w and a bias b, all of them Values with data between -1 and 1."""
        neuron = Neuron(1)
        assert len(neuron.w) == 1
        assert isinstance(neuron.w[0], Value)
        assert -1 <= neuron.w[0].data <= 1
        assert isinstance(neuron.b, Value)
        assert -1 <= neuron.b.data <= 1
