from pathlib import Path
import pandas as pd
from ml.jobs_datasets.raw.category_map import category_map
from progressbar import progressbar
from ml.jobs_datasets.cleaning import is_valid
from ml.utils import get_extension, read_file

FILE_DIR = Path(__file__).parent
OUTPUT_PATH = FILE_DIR / "compiled" / "full.parquet"


def remap_columns(df: pd.DataFrame):
    overlapping_categories = set(category_map.keys()) & set(df.columns)
    return df[list(overlapping_categories)].rename(columns=category_map)


def write_values(key, values):
    with open(FILE_DIR / "compiled" / "full.csv", "w+") as f:
        f.writelines([f'"{key}","{value}"' for value in values])


def compile():
    output_df = pd.DataFrame(dict(label=[], text=[]))

    for path in progressbar(list((FILE_DIR / "raw").iterdir())):
        path = str(path)
        if path.endswith("__pycache__") or get_extension(path) in {"py"}:
            continue
        df = read_file(path)
        df = remap_columns(df)
        for key, series in df._series.items():
            df = pd.DataFrame(
                dict(
                    label=key,
                    text=[value for value in series.values if is_valid(key, value)],
                )
            )
            output_df = output_df.append(df)

    output_df.to_parquet(OUTPUT_PATH)


def load_compiled():
    return pd.read_parquet(OUTPUT_PATH)


if __name__ == "__main__":
    compile()
