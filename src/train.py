"""
train.py
Entrenamiento con MLflow tracking completo.
SOLUCIONA: problemas #4, #5, #6 de la Parte 1
Uso: python src/train.py
"""

import os, sys, yaml, mlflow, mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocessor, load_and_clean, TARGET

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT,'data','raw',
            'WA_Fn-UseC_-Telco-Customer-Churn.csv')

def load_params():
    with open(os.path.join(ROOT,'params.yaml')) as f:
        return yaml.safe_load(f)

def train():
    params = load_params()
    print('\n=== ENTRENAMIENTO CON MLOPS ===')

    df = load_and_clean(DATA_PATH)
    X  = df.drop(columns=[TARGET])
    y  = df[TARGET]
    print(f'Dataset: {len(df)} registros | Churn: {y.mean()*100:.1f}%')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params['test_size'],
        stratify=y,            # misma proporcion en train y test
        random_state=params['random_state'],
    )

    # Pipeline COMPLETO: preprocesador + clasificador
    # El Pipeline garantiza que los mismos pasos se aplican
    # tanto en entrenamiento como en prediccion
    model = Pipeline([
        ('prep', preprocessor),
        ('clf',  RandomForestClassifier(
            n_estimators = params['n_estimators'],
            max_depth    = params['max_depth'],
            class_weight = 'balanced',  # SOLUCION #4: ahora si
            random_state = params['random_state'],
            n_jobs       = -1,
        )),
    ])

    # SOLUCION #5: tracking completo de cada experimento
    mlflow.set_experiment('churn-prediction')

    with mlflow.start_run():
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]

        metrics = {
            'f1':  round(f1_score(y_test, y_pred), 4),
            'auc': round(roc_auc_score(y_test, y_proba), 4),
        }

        # Registrar TODO en MLflow — SOLUCION #5 y #6
        mlflow.log_params(params)          # que parametros
        mlflow.log_metrics(metrics)        # que resultado
        mlflow.sklearn.log_model(          # el modelo completo
            sk_model              = model,
            artifact_path         = 'model',
            registered_model_name = 'churn-rf',
        )

        print(f"F1={metrics['f1']}  AUC={metrics['auc']}")
        print(classification_report(
            y_test, y_pred, target_names=['No Churn','Churn']))
        print('Modelo registrado en MLflow Model Registry')

if __name__ == '__main__':
    train()
