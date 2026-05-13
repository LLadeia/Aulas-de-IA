"""Machine learning model for thermal comfort classification."""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

_model = None
_label_encoder = None


def _train():
    global _model, _label_encoder
    # read dataset
    df = pd.read_csv("dataset.csv")
    X = df[["temperatura", "umidade"]].values
    y = df["classe"].values

    # encode labels
    _label_encoder = LabelEncoder()
    y_enc = _label_encoder.fit_transform(y)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y_enc)
    _model = clf


def predict(temperatura: float, umidade: float) -> str:
    """Predict the class using the trained decision tree.
    Training is performed on first call if necessary.
    """
    global _model, _label_encoder
    if _model is None:
        _train()
    X = [[temperatura, umidade]]
    pred_enc = _model.predict(X)[0]
    return _label_encoder.inverse_transform([pred_enc])[0]
