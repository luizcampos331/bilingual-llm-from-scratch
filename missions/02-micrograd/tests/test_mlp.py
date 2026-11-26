from mlp import MLP
from value import Value


class TestMLP:
    def test_mlp_starts_with_layers_chained_by_size(self):
        """Given MLP(3, [4, 4, 1]), expect 3 layers: the first takes 3 inputs and has 4 neurons, the second takes 4 and has 4, the last takes 4 and has 1."""
        mlp = MLP(3, [4, 4, 1])
        assert len(mlp.layers) == 3
        assert len(mlp.layers[0].neurons) == 4
        assert len(mlp.layers[0].neurons[0].w) == 3
        assert len(mlp.layers[1].neurons) == 4
        assert len(mlp.layers[1].neurons[0].w) == 4
        assert len(mlp.layers[2].neurons) == 1
        assert len(mlp.layers[2].neurons[0].w) == 4

    def test_mlp_output_is_one_value_per_neuron_in_the_last_layer(self):
        """Given MLP(3, [4, 4, 1]), expect mlp([2.0, 3.0, -1.0]) to be a list with 1 Value."""
        mlp = MLP(3, [4, 4, 1])
        result = mlp([2.0, 3.0, -1.0])
        assert len(result) == 1
        assert isinstance(result[0], Value)
