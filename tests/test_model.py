"""
test_model.py — Quality gates del modelo.
SOLUCIONA: problema #8 de la Parte 1
Si alguno falla, el CI/CD bloquea el despliegue automaticamente.
Ejecutar: pytest tests/test_model.py -v
"""

import os, sys, pytest, numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess import load_and_clean, TARGET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

MIN_F1, MIN_AUC = 0.75, 0.80

@pytest.fixture(scope='module')
def model():
    MODEL_URI = (
        "file:///C:/Users/forma_u/Documents/mlopslab/mlruns/1/models/"
        "m-daf3607be85a4404ae5a60a1913f3b14/artifacts"
    )
    return mlflow.sklearn.load_model(MODEL_URI)

@pytest.fixture(scope='module')
def data():
    df = load_and_clean(DATA_PATH)
    return df.drop(columns=[TARGET]), df[TARGET]

def test_f1_minimo(model, data):
    X, y = data
    f1 = f1_score(y, model.predict(X))
    assert f1 >= MIN_F1

def test_auc_minimo(model, data):
    X, y = data
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    assert auc >= MIN_AUC

def test_no_una_sola_clase(model, data):
    X, _ = data
    assert len(set(model.predict(X))) > 1

def test_probabilidades_validas(model, data):
    X, _ = data
    p = model.predict_proba(X.head(100))
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)

def test_determinista(model, data):
    X, _ = data
    s = X.head(10)
    assert (model.predict(s) == model.predict(s)).all()