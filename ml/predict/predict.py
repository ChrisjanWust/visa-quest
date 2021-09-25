import pickle
from sklearn import base
from lazy_object_proxy import Proxy
from ml.features.features import to_features
from ml.settings import MODEL_PATH
from typing import List


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


clf: base.ClassifierMixin = Proxy(load_model)


class ProbaDict(dict):
    def best_class(self):
        return list(self.keys())[0]


def predict_probas(texts) -> List[dict]:
    X = [list(to_features(text).values()) for text in texts]
    proba_sets = clf.predict_proba(X)
    return [
        ProbaDict(
            {
                class_name: proba
                for class_name, proba in sorted(
                    zip(clf.classes_, proba_set),
                    key=lambda tup: tup[0],
                    reverse=True,
                )
            }
        )
        for proba_set in proba_sets
    ]


def predict_best_class(texts) -> List[str]:
    return [proba_dict.best_class() for proba_dict in predict_probas(texts)]


def test_load_model():
    assert isinstance(clf, base.ClassifierMixin)


def test_predict_class():
    assert predict_best_class(["Machine Learning Engineer"]) == ["title"]
