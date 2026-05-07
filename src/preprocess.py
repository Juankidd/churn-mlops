
"""
preprocess.py
Pipeline de preprocesamiento modular.
Se importa desde: train.py, api/app.py, tests/
SOLUCION a los problemas #2 (ruta) y #3 (LabelEncoder) de la Parte 1
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# ── Columnas declaradas explicitamente ───────────────────────
# SOLUCION #3: categorias fijas, no dependen del orden del CSV
NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
]

TARGET = 'Churn'

# ── Limpieza del CSV crudo ───────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    """Carga el CSV y aplica limpieza minima reproducible."""
    df = pd.read_csv(path)                  # path como parametro,
                                            # nunca hardcodeado
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce'
    )
    df[TARGET] = (df[TARGET] == 'Yes').astype(int)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    return df

# ── Pipelines de transformacion ──────────────────────────────
# SOLUCION #3: OrdinalEncoder dentro del Pipeline — el mapping
# se serializa junto con el modelo en MLflow
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1,   # categorias nuevas -> -1, no error
    )),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline,     NUMERIC_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
    ],
    remainder='drop',
)
