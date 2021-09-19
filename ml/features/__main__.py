import pandas as pd
from pathlib import Path
from multiprocessing import Pool
import tqdm

from ml.utils import read_file
from ml.features.features import to_features
from ml import jobs_datasets

OUTPUT_PATH = Path(__file__).parent / "full.parquet"


def to_output_cols(row: tuple):
    return {"label": row[1], **to_features(row[2])}


def main():
    # load dataset
    df = read_file(jobs_datasets.OUTPUT_PATH)
    df = df.sample(frac=1)  # shuffle so load is spread
    # apply features
    print("Applying features")
    tasks = list(map(tuple, df.itertuples()))
    pool = Pool(6)
    df = pd.DataFrame(
        list(tqdm.tqdm(pool.imap_unordered(to_output_cols, tasks), total=len(tasks)))
    )
    pool.close()
    pool.join()
    # with Pool(6) as p:
    #     features = list(p.map(to_output_cols, map(tuple, df.itertuples())))
    # save
    df.to_parquet(OUTPUT_PATH)


if __name__ == "__main__":
    main()
