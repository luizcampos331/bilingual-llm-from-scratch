"""Squared error loss built on Value (Mission 2, micrograd)."""

from collections.abc import Sequence

from value import Value


def squared_error_loss(predictions: Sequence[Value], targets: Sequence[float]) -> Value:
    """Return the sum of (prediction - target) ** 2 over all pairs, as a Value (Value(0) when there are no pairs)."""
    return sum(
        (
            (prediction - target) ** 2
            for prediction, target in zip(predictions, targets)
        ),
        Value(0),
    )
