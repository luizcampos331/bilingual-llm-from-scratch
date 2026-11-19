import pytest
from value import Value


class TestValues:
    def test_value_stores_its_data(self):
        """Given Value(2.0), expect data to be 2.0."""
        value = Value(2.0)
        assert value.data == 2.0

    def test_adding_two_values(self):
        """Given Value(2) + Value(-3), expect a Value with data -1."""
        value1 = Value(2)
        value2 = Value(-3)
        result = value1 + value2
        assert result.data == -1

    def test_multiplying_two_values(self):
        """Given Value(2) * Value(-3), expect a Value with data -6."""
        value1 = Value(2)
        value2 = Value(-3)
        result = value1 * value2
        assert result.data == -6

    def test_paper_expression(self):
        """Given a=2, b=-3, c=10 and f=-2, expect L = (a * b + c) * f to have data -8."""
        a = Value(2)
        b = Value(-3)
        c = Value(10)
        f = Value(-2)
        e = a * b
        d = e + c
        L = d * f
        assert L.data == -8

    def test_grad_starts_at_zero(self):
        """Given a new Value, expect grad to be 0."""
        value = Value(2)
        assert value.grad == 0

    def test_value_remembers_its_parents(self):
        """Given a and b typed by hand and c = a + b, expect c._prev to be {a, b} and a._prev to be empty."""
        a = Value(2)
        b = Value(-3)
        c = a + b
        assert c._prev == {a, b}
        assert a._prev == set()
        assert b._prev == set()


class TestLocalBackward:
    def test_addition_passes_grad_through(self):
        """Given d = e + c with d.grad = 1, expect d._backward() to set e.grad and c.grad to 1."""
        e = Value(-6)
        c = Value(10)
        d = e + c
        d.grad = 1
        d._backward()
        assert e.grad == 1
        assert c.grad == 1

    def test_multiplication_grad_is_the_other_value(self):
        """Given L = d * f with d=4, f=-2 and L.grad = 1, expect L._backward() to set d.grad to -2 (value of f) and f.grad to 4 (value of d)."""
        d = Value(4)
        f = Value(-2)
        L = d * f
        L.grad = 1
        L._backward()
        assert d.grad == -2
        assert f.grad == 4

    def test_tanh_value_and_grad(self):
        """Given n = 0.8814, expect o = n.tanh() to be about 0.7071, and with o.grad = 1, expect o._backward() to set n.grad to about 0.5 (1 - o**2)."""
        n = Value(0.8814)
        o = n.tanh()
        assert o.data == pytest.approx(0.7071, abs=0.0001)
        o.grad = 1
        o._backward()
        assert n.grad == pytest.approx(0.5, abs=0.0001)


class TestBackward:
    def test_backward_fills_paper_expression(self):
        """Given L = (a * b + c) * f with a=2, b=-3, c=10, f=-2, expect L.backward() to set grads a=6, b=-4, c=-2, e=-2, d=-2, f=4 and L=1."""
        a = Value(2)
        b = Value(-3)
        c = Value(10)
        f = Value(-2)
        e = a * b
        d = e + c
        L = d * f
        L.backward()
        assert a.grad == 6
        assert b.grad == -4
        assert c.grad == -2
        assert f.grad == 4
        assert e.grad == -2
        assert d.grad == -2
        assert L.grad == 1
