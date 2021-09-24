import pandas as pd
from pathlib import Path
from lazy_object_proxy import Proxy
from functools import cached_property
import math

from ml.utils import read_file


PATH = Path(__file__).parent / "word_frequencies.csv"


def load_word_frequencies(limit=150_000):
    df = read_file(PATH, index_col=0)
    # limit words - some garbage towards the end
    df = df[:limit]
    # convert to frequency
    df["frequency"] = df["count"] / sum(df["count"])
    # add percentile
    df["rank"] = df["count"].rank(ascending=False).astype(int)
    df["percentile"] = df["count"].rank(pct=True)
    return df


class _WordFreq:
    # name is private - use word_freq rather

    data: pd.DataFrame = Proxy(load_word_frequencies)

    @cached_property
    def corpus_size(self):
        return len(self.data)

    @cached_property
    def ranks(self):
        return {key: int(row["rank"]) for key, row in self.data.iterrows()}

    def rank(self, word):
        return self.ranks[word]

    def histogram(self, text: str, base=2):
        word_percentiles = []
        words = text.split(" ")
        for word in words:
            word = word.strip().lower()  # todo: clean word better
            if word in self.data.index:
                word_percentiles.append(self.rank(word))
            else:
                word_percentiles.append(self.corpus_size * base)

        word_percentiles = sorted(word_percentiles)
        counts = []
        for i in range(0, math.ceil(math.log(self.corpus_size, base) + 2)):
            counts.append(0)
            while word_percentiles and word_percentiles[0] <= base ** i:
                counts[-1] += 1 / len(words)
                del word_percentiles[0]

        return counts


word_freq = _WordFreq()

if __name__ == "__main__":
    word_frequencies = _WordFreq()
    text = "the man is medium giberishgiberish"
    res = word_frequencies.histogram(text)
    print(res)
    assert res[0]
    assert res[-1]
