import random

import pytest
from bigram import END, START, Bigram


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
        """Given "barbara" with +1 smoothing, expect the row for "a" to be ar = 3/7 and a<e> = 2/7 (pytest.approx)."""
        bigram = Bigram(["barbara"])
        assert bigram.probability("a", "r") == pytest.approx(3 / 7)
        assert bigram.probability("a", END) == pytest.approx(2 / 7)

    def test_unseen_pair_has_nonzero_probability(self):
        """Given "barbara" with +1 smoothing, expect the unseen pair ab to be (0 + 1) / (3 + 4) = 1/7."""
        bigram = Bigram(["barbara"])
        assert bigram.probability("a", "b") == pytest.approx(1 / 7)

    def test_every_row_with_counts_sums_to_one(self):
        """Given "barbara", expect every row with counts (<s>, b, a, r) to sum to 1."""
        bigram = Bigram(["barbara"])
        for first in [START, "b", "a", "r"]:
            total = sum(bigram.row(first).values())
            assert total == pytest.approx(1)


class TestSampling:
    def test_sample_after_a_is_always_in_row(self):
        """Given "barbara" and a seeded generator, expect every sample after "a" to be one of the keys of row("a") (b, a, r, <e>)."""
        bigram = Bigram(["barbara"])
        row = bigram.row("a")
        rng = random.Random(42)
        for _ in range(100):
            sample = bigram.sample_next("a", rng)
            assert sample in row
