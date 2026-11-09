import pytest
from bigram import END, Bigram


class TestCounting:
    def test_barbara_has_eight_pairs(self):
        """Given "barbara", expect 8 pairs, including START and END."""
        bigram = Bigram(["barbara"])
        pairs = bigram.bigram_count_pairs
        assert sum(pairs.values()) == 8

    def test_barbara_counts_ba_and_ar_twice(self):
        """Given "barbara", expect "ba" and "ar" to each have count 2."""
        bigram = Bigram(["barbara"])
        assert bigram.count("b", "a") == 2
        assert bigram.count("a", "r") == 2

    def test_counting_ignores_case(self):
        """Given "Barbara", expect the same counts as "barbara"."""
        bigram = Bigram(["Barbara"])
        assert bigram.count("B", "a") == 2
        assert bigram.count("b", "A") == 2
        assert bigram.count("A", "r") == 2
        assert bigram.count("a", "R") == 2


class TestProbabilities:
    def test_barbara_row_for_a_has_expected_probabilities(self):
        """Given "barbara", expect the row for "a" to be ar = 2/3 and a<e> = 1/3 (pytest.approx)."""
        bigram = Bigram(["barbara"])
        assert bigram.probability("a", "r") == pytest.approx(2 / 3)
        assert bigram.probability("a", END) == pytest.approx(1 / 3)
