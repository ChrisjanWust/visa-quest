import re
import pandas as pd
from pathlib import Path
from progressbar import progressbar

from ml.utils import read_file
from ml import jobs_datasets

OUTPUT_PATH = Path(__file__).parent / "full.parquet"
df = read_file(jobs_datasets.OUTPUT_PATH)


#%%  occurrence of patterns across different labels
def get_pattern_perc(label, pattern):
    all_text = "".join(df[df.label == label].text)
    return len(re.findall(pattern, all_text)) / len(all_text)


data = [
    {
        "label": label,
        **{
            pattern: get_pattern_perc(label, pattern)
            for pattern in [r"/", r"\.", r"\(", r"-", "[A-Z]"]
        },
    }
    for label in progressbar(set(df.label))
]

table_data = pd.DataFrame(data)
print(table_data)

#%%
