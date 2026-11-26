"""Squared error loss built on Value (Mission 2, micrograd)."""


def squared_error_loss(predictions, targets):
    """Return the sum of (prediction - target) ** 2 over all pairs, as a Value."""
    return sum(
        (prediction - target) ** 2 for prediction, target in zip(predictions, targets)
    )
