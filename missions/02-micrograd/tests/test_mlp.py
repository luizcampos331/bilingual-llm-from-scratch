from mlp import MLP


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
