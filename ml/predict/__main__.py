import pickle
import sklearn.base
from lazy_object_proxy import Proxy
from features.features import to_features
from ml.training.__main__ import MODEL_PATH


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


clf: sklearn.tree.DecisionTreeClassifier = Proxy(load_model)


def predict_class(texts):
    X = [list(to_features(text).values()) for text in texts]
    proba_sets = clf.predict_proba(X)
    return [
        {
            class_name: proba
            for class_name, proba in sorted(
                zip(proba_set, clf.classes_), key=lambda tup: tup[0]
            )
        }
        for proba_set in proba_sets
    ]


def test_load_model():
    assert isinstance(clf, sklearn.base.ClassifierMixin)


def test_predict_class():
    assert predict_class(["Machine Learning Engineer"] == "title")
