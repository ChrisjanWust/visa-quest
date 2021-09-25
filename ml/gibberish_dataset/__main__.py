#%%
import os
from dataclasses import dataclass
from pathlib import Path
import gzip
from scraping.consolidator.element_annotator import extract_elements_from_text
import csv
import pandas as pd
from typing import Union
import random

from ml.settings import CACHE_DIR_PATH, GIBBERISH_CSV_PATH


assert os.environ["PYTHONHASHSEED"] == "0"


def shuffle_and_return(iterable: iter):
    lst = list(iterable)
    random.shuffle(lst)
    return lst


@dataclass
class IndexedElement:
    main_key: str
    page_key: str
    position: int
    text: str
    previous_text: str
    nr_ancestors: int
    nr_children: int
    nr_descendants: int
    tag: str
    child_tags: list
    is_gibberish: bool
    text_hash: int = None
    key: str = None

    def __post_init__(self):
        if not self.key:
            self.key = self.generate_key()
        if not self.text_hash:
            self.text_hash = hash(self.text)

    def serialize(self):
        data = self.__dict__
        data["child_tags"] = ",".join(self.child_tags)
        return {
            "key": self.key,
            **data,
        }

    def serialize_row(self):
        return tuple(self.serialize().values())

    def generate_key(self):
        return f"{self.main_key}-{self.page_key}-{self.position}-{self.text_hash}"

    @classmethod
    def all_fields(cls):
        return ["key"] + list(cls.__dataclass_fields__.keys())[:-1]


def write_row(row: Union[list, tuple]):
    with open(GIBBERISH_CSV_PATH, "a+") as f:
        csv.writer(f).writerow(row)


file_existed = GIBBERISH_CSV_PATH.exists()

last_row = None
done_keys = set()
previous_decisions = {}

if not file_existed:
    write_row(IndexedElement.all_fields())
else:
    df = pd.read_csv(GIBBERISH_CSV_PATH)[
        ["main_key", "page_key", "text_hash", "key", "is_gibberish"]
    ]
    done_keys = {tuple(row) for row in df[["main_key", "page_key"]].values}
    previous_decisions = {
        text_hash: is_gibberish
        for text_hash, is_gibberish in df[["text_hash", "is_gibberish"]].values
    }
    last_row = df.iloc[-1]
    del df


#%%
for main_folder in shuffle_and_return(CACHE_DIR.iterdir()):
    main_key = main_folder.name
    for page_folder in shuffle_and_return(main_folder.iterdir()):
        page_key = page_folder.name
        in_last_row = (
            # there was previous data in the file
            last_row is not None
            # and we've looked at this page already (note: perhaps not the whole page!)
            and last_row.main_key == main_key
            and last_row.page_key == page_key
        )
        if (main_folder.name, page_folder.name) in done_keys and not in_last_row:
            continue
        with gzip.open(page_folder / "response_body", "rb") as f:
            html_body = f.read()
            for element in extract_elements_from_text(html_body):
                # skip if already written to csv
                if in_last_row and element["position"] < last_row.position:
                    continue

                text = element["text"]
                text_hash = hash(text)
                if text_hash in previous_decisions:  # skip already seen texts
                    is_gibberish = previous_decisions[text_hash]
                    print(f"Memorised {text[:10]}..., saving and moving on.")
                else:  # get manual classification
                    print("-----------------------------------------------------------")
                    print(text)
                    print("-----------------------------------------------------------")
                    input_str = input("is_gibberish:").lower()[:1]
                    is_gibberish = input_str not in ["f", "n", "0"]
                    print("is_gibberish:", is_gibberish)

                previous_decisions[text_hash] = is_gibberish
                indexed_element = IndexedElement(
                    main_key=main_key,
                    page_key=page_key,
                    is_gibberish=is_gibberish,
                    text_hash=text_hash,
                    **element,
                )
                write_row(indexed_element.serialize_row())
