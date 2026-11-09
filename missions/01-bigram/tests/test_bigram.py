from bigram import Bigram


class TestCounting:
    def test_barbara_has_eight_pairs(self):
        bigram = Bigram(["barbara"])
        pairs = bigram.pairs
        assert len(pairs) == 8
