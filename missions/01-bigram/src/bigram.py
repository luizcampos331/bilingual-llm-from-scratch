"""Character-level bigram language model (Mission 1)."""

import math
import random

START = "<s>"
END = "<e>"


class Bigram:
    """Character-level bigram model built from a list of names."""

    def __init__(self, names: list[str]):
        """Count character pairs from the given names, wrapping each name with START and END, and collect the possible next characters."""
        self.bigram_count_pairs = {}
        self.next_chars = set()
        for name in names:
            for bigram in self._pairs(name):
                self.bigram_count_pairs[bigram] = (
                    self.bigram_count_pairs.get(bigram, 0) + 1
                )
                self.next_chars.add(bigram[1])

    def _pairs(self, name: str) -> list[tuple[str, str]]:
        """Return the consecutive character pairs of name, lowercased and wrapped with START and END."""
        chars = [START] + list(name.lower()) + [END]
        return [(chars[i], chars[i + 1]) for i in range(len(chars) - 1)]

    def count(self, first_char: str, second_char: str) -> int:
        """Return how many times second_char followed first_char in the training names."""
        return self.bigram_count_pairs.get((first_char.lower(), second_char.lower()), 0)

    def probability(self, first_char: str, second_char: str) -> float:
        """Return P(second_char | first_char) with +1 smoothing: (count + 1) / (row total + number of possible next characters)."""
        denominator = sum(
            value
            for (first, _), value in self.bigram_count_pairs.items()
            if first == first_char.lower()
        ) + len(self.next_chars)
        return (self.count(first_char, second_char) + 1) / denominator

    def row(self, first_char: str) -> dict[str, float]:
        """Return the distribution of next characters after first_char, as {second_char: P(second_char | first_char)}."""
        row_dict = {}
        for ch in sorted(self.next_chars):
            row_dict[ch] = self.probability(first_char, ch)
        return row_dict

    def sample_next(self, first_char: str, rng: random.Random) -> str:
        """Draw the next character after first_char, weighted by the row of first_char."""
        row = self.row(first_char)
        return rng.choices(list(row.keys()), weights=list(row.values()))[0]

    def loss(self, names: list[str]) -> float:
        """Return the average negative log-likelihood (mean of -ln P) over every pair of the given names, wrapped with START and END."""
        total_loss = 0
        total_pairs = 0
        for name in names:
            for first, second in self._pairs(name):
                total_loss += -math.log(self.probability(first, second))
                total_pairs += 1
        return total_loss / total_pairs
