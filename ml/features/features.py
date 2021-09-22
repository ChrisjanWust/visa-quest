import re
import nltk
from collections import Counter

ROLE_MIN_LEN = 7
ROLE_ENDINGS = {"ent", "ian", "ant", "ist", "er", "or", "yst", "ive"}
SYMBOLS = [r"/", r"\.", r"\(", r"-", "[A-Z]"]
NLTK_TAGS = list(nltk.data.load("help/tagsets/upenn_tagset.pickle"))

symbol_patterns = [re.compile(pattern) for pattern in SYMBOLS]
role_pattern = re.compile(
    "|".join([f"({ending[::-1]})" for ending in ROLE_ENDINGS]),
    flags=re.IGNORECASE,
)


def to_features(text: str):
    return {
        "len": len(text),
        "perc_digits": len([c for c in text if c.isdigit()]) / len(text),
        "perc_role_words": perc_words_ending_in_role(text),
        **word_tags(text),
        **symbol_counts(text),
    }


def perc_words_ending_in_role(text):
    words = text.split()
    return len([word for word in words if word_ends_in_role(word)]) / len(words)


def word_ends_in_role(word: str):
    if len(word) < ROLE_MIN_LEN:
        return False
    return bool(role_pattern.match(word))


def word_tags(text):
    words = text.split()
    tags = [tag for word, tag in nltk.pos_tag(words)]
    counts = Counter(tags)
    return {tag: counts.get(tag, 0) / len(words) for tag in NLTK_TAGS}


def symbol_counts(text):
    return {
        pattern.pattern: len(pattern.findall(text)) / len(text)
        for pattern in symbol_patterns
    }
