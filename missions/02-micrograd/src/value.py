"""Scalar value with automatic gradient computation (Mission 2, micrograd)."""


class Value:
    """A single number that remembers how it was computed, so its gradient can be found later."""

    def __init__(self, data: float):
        """Store data, start grad at 0, and start with a _backward that does nothing (a value typed by hand has no inputs to pass grad to)."""
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None

    def __add__(self, other: "Value") -> "Value":
        """Return a new Value whose data is self.data + other.data."""
        out = Value(self.data + other.data)

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: "Value") -> "Value":
        """Return a new Value whose data is self.data * other.data."""
        out = Value(self.data * other.data)

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
