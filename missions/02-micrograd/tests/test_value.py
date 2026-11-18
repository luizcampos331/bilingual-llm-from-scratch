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
