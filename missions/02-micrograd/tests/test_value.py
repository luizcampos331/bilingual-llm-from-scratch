from value import Value


class TestValues:
    def test_value_stores_its_data(self):
        """Given Value(2.0), expect data to be 2.0."""
        value = Value(2.0)
        assert value.data == 2.0
