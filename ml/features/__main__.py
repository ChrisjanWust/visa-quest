import pandas as pd
from pathlib import Path
from multiprocessing import Pool
import tqdm
from copy import deepcopy

from ml.utils import read_file
from ml.features.features import to_features
from ml import jobs_datasets
from ml import settings


OUTPUT_PATH = Path(__file__).parent / "full.parquet"


# todo: data cleaning: remove some samples because they're invalid. See short descriptions.


def to_output_cols(row: tuple):
    return {"label": row[1], **to_features(row[2])}


to_output_cols = deepcopy(to_output_cols)


def filter_invalid_samples(df):
    df = df.reset_index(drop=True)
    # drop short descriptions. Includes samples like "Location: Dallas, TX" and "Enter Your Email Address"
    short_descriptions = df[(df.label == "description") & (df.text.str.len() < 30)]
    df = df.drop(short_descriptions.index)
    # drop titles longer than 80 chars
    very_long_titles = df[df.label == "title"][df.text.str.len() > 80]
    df = df.drop(very_long_titles.index)
    # was planning to sample titles with medium length, but in practice, it's not necessary
    # long_titles = df[df.label=="title"][df.text.str.len() > 50]
    # df = df.drop(mediumlong_titles.sample(frac=1)[:int(len(df[df.label=="title"])/10*9)]
    # todo: location
    return df


def load_combined_dataset():
    # load dataset
    df_real = read_file(jobs_datasets.OUTPUT_PATH)
    raw_gibberish_df = read_file(settings.GIBBERISH_CSV_PATH)[["is_gibberish", "text"]]
    gibberish_text = list(raw_gibberish_df[raw_gibberish_df.is_gibberish == True].text)
    max_label_count = df_real.label.value_counts().max()
    processed_gibberish_df = pd.DataFrame(
        dict(
            text=list(gibberish_text) * int(max_label_count / len(gibberish_text)),
            label="gibberish",
        )
    )
    return df_real.append(processed_gibberish_df)


def main():
    df = load_combined_dataset()
    df = df.sample(frac=1)  # shuffle so load is spread
    # apply features
    print("Applying features")
    tasks = list(map(tuple, df.itertuples()))
    pool = Pool(6)
    df = pd.DataFrame(
        list(
            tqdm.tqdm(
                pool.imap_unordered(to_output_cols, tasks),
                total=len(tasks),
            )
        )
    )
    pool.close()
    pool.join()
    # with Pool(6) as p:
    #     features = list(p.map(to_output_cols, map(tuple, df.itertuples())))
    # save
    df.to_parquet(OUTPUT_PATH)


#%%
if __name__ == "__main__":
    main()
