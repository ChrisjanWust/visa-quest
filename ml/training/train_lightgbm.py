import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ml.utils import read_file
from ml.features.__main__ import OUTPUT_PATH


# todo: balance dataset


df = read_file(OUTPUT_PATH)

# label encoding
label_encoder = LabelEncoder()
df.label = label_encoder.fit_transform(df.label)


def rename_col(col: str):
    return "".join(c if c.isalnum() else str(ord(c)) for c in col)


# df = df.rename(columns=rename_col)
df = df[[col for col in df.columns if all(c.isalnum() for c in col)]]

# split
def split_three_way(df, perc_2, perc_3):
    split_1, split_2_and_3 = train_test_split(df, test_size=perc_2 + perc_3)
    split_2, split_3 = train_test_split(
        split_2_and_3,
        test_size=perc_3 / (perc_2 + perc_3),
    )
    return split_1, split_2, split_3


df_train_x, df_vali_x, df_test_x = split_three_way(df, 0.2, 0.2)
df_train_y = df_train_x.pop("label")
df_vali_y = df_vali_x.pop("label")
df_test_y = df_test_x.pop("label")

train_data = lgb.Dataset(df_train_x, label=df_train_y)
vali_data = train_data.create_valid(df_vali_x, label=df_vali_y)
test_data = train_data.create_valid(df_test_x, label=df_test_y)

del df_train_x, df_train_y, df_vali_x, df_vali_y, df_test_x, df_test_y

#%%
param = {
    "num_leaves": 120,
    "objective": "multiclass",
    "num_class": 5,
    "metric": ["multi_logloss", "multi_error"],
    # "eval": "multi_error",
    "seed": 0,
    "num_threads": 8,
    "deterministic": True,
    "learning_rate": 0.06,
    "feature_fraction": 0.9,
}
bst = lgb.train(
    param,
    train_set=train_data,
    num_boost_round=200,
    valid_sets=[vali_data],
    keep_training_booster=True,
)
print("Training data")
print(bst.eval(train_data, ""))
print("Validation data")
print(bst.eval(vali_data, ""))
print("Testing data")
print(bst.eval(test_data, ""))
