from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ml.utils import read_file
from ml.features.__main__ import OUTPUT_PATH


# todo: balance dataset


df = read_file(OUTPUT_PATH)


def rename_col(col: str):
    return "".join(c if c.isalnum() else str(ord(c)) for c in col)


df = df.rename(columns=rename_col)

# split
df_train_x, df_test_x = train_test_split(df, test_size=0.2)
df_train_y = df_train_x.pop("label")
df_test_y = df_test_x.pop("label")

#%%
clf = DecisionTreeClassifier()

print(clf)
clf.fit(df_train_x, df_train_y)

score = clf.score(df_train_x, df_train_y)
print(score)
score = clf.score(df_test_x, df_test_y)
print(score)
