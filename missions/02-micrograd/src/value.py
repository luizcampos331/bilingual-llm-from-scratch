"""Scalar value with automatic gradient computation (Mission 2, micrograd)."""


class Value:
    """A single number."""

    def __init__(self, data: float):
        """Store data."""
        self.data = data
