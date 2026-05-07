"""
app.py — API REST del modelo de churn.
SOLUCIONA: problema #7 de la Parte 1 (modelo inaccesible)
Endpoints: GET / | POST /predict | POST /predict/batch
"""

import os, sys, time, logging
from contextlib import asynccontextmanager
from typing import List

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_URI = os.getenv(
    "MODEL_URI",
    "file:///C:/Users/forma_u/Documents/mlopslab/mlruns/1/models/m-daf3607be85a4404ae5a60a1913f3b14/artifacts"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Cargando modelo desde: {MODEL_URI}")

    model = mlflow.pyfunc.load_model(MODEL_URI)

    logger.info(f"Modelo cargado tipo: {type(model)}")

    if model is None:
        raise RuntimeError("El modelo cargó como None")

    app.state.model = model
    logger.info("Modelo listo para predicciones.")

    yield

app = FastAPI(
    title='Churn Predictor API',
    description='Prediccion de abandono de clientes.',
    version='1.0.0',
    lifespan=lifespan,
)

# ── Validacion automatica de entrada (SOLUCION a #3 y #7) ───
class CustomerFeatures(BaseModel):
    tenure:           int   = Field(..., ge=0, le=120)
    MonthlyCharges:   float = Field(..., ge=0)
    TotalCharges:     float = Field(..., ge=0)
    gender:           str
    SeniorCitizen:    int   = Field(..., ge=0, le=1)
    Partner:          str
    Dependents:       str
    PhoneService:     str
    MultipleLines:    str
    InternetService:  str
    OnlineSecurity:   str
    OnlineBackup:     str
    DeviceProtection: str
    TechSupport:      str
    StreamingTV:      str
    StreamingMovies:  str
    Contract:         str
    PaperlessBilling: str
    PaymentMethod:    str

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction:  bool
    risk_level:        str
    latency_ms:        float

def risk_level(prob: float) -> str:
    if prob >= 0.70: return 'high'
    if prob >= 0.40: return 'medium'
    return 'low'

@app.get('/', tags=['Health'])
def health(): return {'status':'ok','service':'churn-predictor'}

@app.post('/predict', response_model=PredictionResponse)
async def predict(customer: CustomerFeatures):
    t0 = time.perf_counter()
    try:
        df   = pd.DataFrame([customer.model_dump()])
        pred = app.state.model.predict(df)
        prob = float(pred[0])
        return PredictionResponse(
            churn_probability = round(prob,4),
            churn_prediction  = prob >= 0.5,
            risk_level        = risk_level(prob),
            latency_ms        = round((time.perf_counter()-t0)*1000,2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class BatchRequest(BaseModel):
    customers: List[CustomerFeatures]

@app.post('/predict/batch')
async def predict_batch(req: BatchRequest):
    t0    = time.perf_counter()
    df    = pd.DataFrame([c.model_dump() for c in req.customers])
    probs = app.state.model.predict(df)
    return {'predictions':[{'churn_probability':round(float(p),4),
             'churn_prediction':float(p)>=0.5,'risk_level':risk_level(float(p))}
            for p in probs],'total':len(probs),
            'latency_ms':round((time.perf_counter()-t0)*1000,2)}
