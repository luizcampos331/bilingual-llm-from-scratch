"""Scalar value with automatic gradient computation (Mission 2, micrograd)."""


class Value:
    """A single number that remembers how it was computed, so its gradient can be found later."""

    def __init__(self, data: float):
        """Store data and start grad at 0."""
        self.data = data
        self.grad = 0.0

    def __add__(self, other: "Value") -> "Value":
        """Return a new Value whose data is self.data + other.data."""
        return Value(self.data + other.data)

    def __mul__(self, other: "Value") -> "Value":
        """Return a new Value whose data is self.data * other.data."""
        return Value(self.data * other.data)
