from ml.utils.word_frequencies import load_word_frequencies, _WordFreq


def test_load_word_frequencies():
    df = load_word_frequencies()
    assert (
        df.loc["a"].percentile
        > df.loc["animal"].percentile
        > df.loc["animalistic"].percentile
    )


def test_word_frequencies():
    word_freq = _WordFreq()
    previous_len = 999
    for base in sorted([2, 3, 10]):
        bins = word_freq.histogram("some words and giberishgiberish", base=base)
        assert not bins[0], "First bin populated, but text does not contain 'the'"
        assert 0.99 < sum(bins) < 1.01
        assert bins[-1], "Last bin empty, but text does contain actual giberish"
        new_len = len(bins)
        assert new_len < previous_len, "Bigger base numbers should have less bins"
        previous_len = new_len


if __name__ == "__main__":
    test_load_word_frequencies()
    test_word_frequencies()
