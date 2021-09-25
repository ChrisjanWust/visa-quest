from functools import partial
from typing import Union
import pandas as pd
from pathlib import Path


def get_extension(path: str):
    name, extension = path.rsplit(".", 1)
    if extension in {"gz", "zip"}:
        return get_extension(name)
    return extension


def read_file(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    if isinstance(path, Path):
        path = str(path)
    read_func_mapping = {
        "csv": pd.read_csv,
        "json": pd.read_json,
        "ldjson": partial(pd.read_json, lines=True),
        "parquet": pd.read_parquet,
    }
    read_function = read_func_mapping[get_extension(path)]
    return read_function(path, **kwargs)
