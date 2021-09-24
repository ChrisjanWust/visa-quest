import re
import nltk
from collections import Counter

from ml.utils.word_frequencies import word_freq

ROLE_MIN_LEN = 7
ROLE_ENDINGS = {"ent", "ian", "ant", "ist", "er", "or", "yst", "ive"}
SYMBOLS = ["/", r"\.", r"\(", "-", "\b[A-Z]", "\n", "\t", ";", "…", r"\|", r"$"]
NLTK_TAGS = list(nltk.data.load("help/tagsets/upenn_tagset.pickle"))

symbol_patterns = [re.compile(pattern) for pattern in SYMBOLS]
role_pattern = re.compile(
    "|".join([f"({ending[::-1]})" for ending in ROLE_ENDINGS]),
    flags=re.IGNORECASE,
)


def clean_text(text: str):
    return text.strip()  # todo: this should be expanded


def to_features(text: str):
    text = clean_text(text)
    return {
        "len": len(text),
        "perc_digits": len([c for c in text if c.isdigit()]) / len(text),
        "perc_role_words": perc_words_ending_in_role(text),
        **word_tags(text),
        **symbol_frequencies(text),
        **word_frequency_bins(text),
    }


def perc_words_ending_in_role(text):
    words = text.split()
    return len([word for word in words if word_ends_in_role(word)]) / (len(words) or 1)


def word_ends_in_role(word: str):
    if len(word) < ROLE_MIN_LEN:
        return False
    return bool(role_pattern.match(word))


def word_tags(text: str):
    words = text.split()
    tags = [tag for word, tag in nltk.pos_tag(words)]
    counts = Counter(tags)
    return {tag: counts.get(tag, 0) / len(words) for tag in NLTK_TAGS}


def symbol_frequencies(text):
    return {
        pattern.pattern: len(pattern.findall(text)) / (text.count(" ") + 1)
        for pattern in symbol_patterns
    }


def word_frequency_bins(text):
    bins = word_freq.histogram(text)
    return {f"bin_freq_{i}": bin_ for i, bin_ in enumerate(bins)}


# todo: dynamically get top X words of each category and tokenize them
# thinking X should be around 5 to avoid model bias
