import sklearn.base

from features.features import to_features

clf: sklearn.tree.DecisionTreeClassifier


def predict_class(texts):
    X = [list(to_features(text).values()) for text in texts]
    probas = clf.predict_proba(X)
    return [(clf.classes_[proba.argmax()], proba.max()) for proba in probas]


predictions = clf.predict_proba(
    [list(to_features(text).values()) for text in ["Machine Learning Engineer"]]
)


predict_class("Title: Machine Learning Engineer")
