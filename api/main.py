from pathlib import Path

import joblib
from fastapi import FastAPI
import pandas as pd

from .schemas import Budget

# chemin vers models/
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" # ../demo-regression/models

# chargement des modèles
model = joblib.load(MODELS_DIR / "model_scaled.joblib")
scaler = joblib.load(MODELS_DIR / "scaler.joblib")

# instanciation de FastAPI
app = FastAPI(title="API de prédiction des ventes")

#### API ####

@app.get("/")
def accueil():

    return {"message": "API ok"}

@app.post("/predict")
def predict(budgets: Budget):
    print("Données front : ", budgets)
    X = pd.DataFrame(
        [[budgets.TV, budgets.radio, budgets.newspaper]],
        columns=["TV", "radio", "newspaper"]
    )
    X_scaled = scaler.transform(X)
    sales = model.predict(X_scaled)[0]
    return {"sales": round(float(sales), 2)}