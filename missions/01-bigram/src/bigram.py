"""Character-level bigram language model (Mission 1)."""

START = "<s>"
END = "<e>"


class Bigram:
    """Character-level bigram model built from a list of names."""

    def __init__(self, names: list[str]):
        """Build the list of character pairs from the given names, wrapping each name with START and END.

        Args:
            names: List of names to build the pairs from.
        """
        self.pairs = []
        for name in names:
            name = [START] + list[str](name) + [END]
            for i in range(len(name) - 1):
                self.pairs.append((name[i], name[i + 1]))
