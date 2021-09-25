from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle  # todo: use joblib

from ml.utils import read_file
from ml.features.__main__ import OUTPUT_PATH
from ml.settings import MODEL_PATH


# todo: balance dataset
"""
Actually, just balance the testing set. Reason being is the data is well balance, 
except for the salary. Also, salary is scoring really well, perhaps the best (98.3%).
So reducing total samples by x50 to balance isn't worth it.
Might not even be necessary.
"""


df = read_file(OUTPUT_PATH)


def rename_col(col: str):
    return "".join(c if c.isalnum() else str(ord(c)) for c in col)


df = df.rename(columns=rename_col)

# split
df_train_x, df_test_x = train_test_split(df, test_size=0.2)
df_train_y = df_train_x.pop("label")
df_test_y = df_test_x.pop("label")

#%%
# clf = DecisionTreeClassifier()
clf = RandomForestClassifier(
    n_jobs=7,
    verbose=2,
    n_estimators=120,
    min_samples_split=0.00001,
    bootstrap=True,
    max_samples=0.7,
    # ccp_alpha=0.01,
)

print(f"Training {clf}")
clf.fit(df_train_x, df_train_y)

score = clf.score(df_train_x, df_train_y)
print("train:", score)
score = clf.score(df_test_x, df_test_y)
print("test", score)

with open(MODEL_PATH, "wb+") as f:
    pickle.dump(clf, f)

#%%

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def get_max_err(conf_matrix) -> int:
    """
    Return value of max error cell from confusion matrix
    """
    return max(
        [
            value
            for j, row in enumerate(conf_matrix)
            for i, value in enumerate(row)
            if i != j
        ]
    )


conf_matrix = confusion_matrix(
    y_true=df_test_y, y_pred=clf.predict(df_test_x), labels=clf.classes_
)


#%%
df_cm = pd.DataFrame(conf_matrix, clf.classes_, clf.classes_)
sn.heatmap(
    df_cm,
    vmax=1.5 * get_max_err(conf_matrix),
    annot=True,
    annot_kws={"size": 10},
    # norm=LogNorm(),
    fmt="g",
    square=True,
)

plt.show()
