"""Scalar value with automatic gradient computation (Mission 2, micrograd)."""

import math


class Value:
    """A single number that remembers how it was computed, so its gradient can be found later."""

    def __init__(self, data: float, _children=()):
        """Store data, start grad at 0, and start with a _backward that does nothing (a value typed by hand has no inputs to pass grad to)."""
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other: "Value | float") -> "Value":
        """Return a new Value whose data is self.data + other.data."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        """Return a new Value whose data is self.data * other.data."""
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: "Value | float") -> "Value":
        """Return a new Value whose data is self.data - other.data."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other))
        return out

    def __rmul__(self, other: "Value | float") -> "Value":
        """Called by Python for 3 * a (number on the left): multiplication order does not matter, so return self * other."""
        return self * other

    def tanh(self) -> "Value":
        """Return a new Value whose data is tanh(self.data), a number squashed into the range -1 to 1."""
        out = Value(math.tanh(self.data), (self,))

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Fill grad for every Value that led to this one, starting with self.grad = 1 and walking the graph backwards in topological order."""
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build(parent)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
