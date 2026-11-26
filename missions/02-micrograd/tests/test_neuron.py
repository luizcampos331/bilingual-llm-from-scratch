import pytest
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

    def test_neuron_output_with_weights_set_by_hand(self):
        """Given Neuron(1) with w[0] = -3 and b = 6.8814, expect neuron([2]) to be a Value close to 0.7071 (tanh of 2 * -3 + 6.8814 = 0.8814)."""
        neuron = Neuron(1)
        neuron.w[0] = Value(-3)
        neuron.b = Value(6.8814)
        result = neuron([2])
        assert isinstance(result, Value)
        assert result.data == pytest.approx(0.7071, abs=1e-4)
