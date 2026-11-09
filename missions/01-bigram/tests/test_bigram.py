from bigram import Bigram


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
