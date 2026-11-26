from loss import squared_error_loss
from value import Value


class TestSquaredErrorLoss:
    def test_loss_is_zero_when_predictions_match_targets(self):
        """Given predictions [Value(1), Value(-1)] and targets [1, -1], expect the loss to be a Value with data 0."""
        predictions = [Value(1), Value(-1)]
        targets = [1, -1]
        loss = squared_error_loss(predictions, targets)
        assert isinstance(loss, Value)
        assert loss.data == 0

    def test_loss_with_values_calculated_by_hand(self):
        """Given predictions [Value(0.5), Value(-0.5)] and targets [1, -1], expect loss data 0.5 ((0.5 - 1)² + (-0.5 - (-1))² = 0.25 + 0.25)."""
        predictions = [Value(0.5), Value(-0.5)]
        targets = [1, -1]
        loss = squared_error_loss(predictions, targets)
        assert isinstance(loss, Value)
        assert loss.data == 0.5
