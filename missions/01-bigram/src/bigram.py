"""Character-level bigram language model (Mission 1)."""

START = "<s>"
END = "<e>"


class Bigram:
    """Character-level bigram model built from a list of names."""

    def __init__(self, names: list[str]):
        """Count character pairs from the given names, wrapping each name with START and END."""
        self.bigram_count_pairs = {}
        for name in names:
            name_lower = name.lower()
            name = [START] + list(name_lower) + [END]
            for i in range(len(name) - 1):
                bigram = (name[i], name[i + 1])
                self.bigram_count_pairs[bigram] = (
                    self.bigram_count_pairs.get(bigram, 0) + 1
                )

    def count(self, first_char: str, second_char: str) -> int:
        """Return how many times second_char followed first_char in the training names."""
        return self.bigram_count_pairs.get((first_char.lower(), second_char.lower()), 0)

    def probability(self, first_char: str, second_char: str) -> float:
        """Return P(second_char | first_char): the count of (first_char, second_char) divided by the total of row first_char."""
        denominator = sum(
            value
            for (first, _), value in self.bigram_count_pairs.items()
            if first == first_char.lower()
        )
        return self.count(first_char, second_char) / denominator
