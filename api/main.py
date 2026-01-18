from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialisation de l'API
app = FastAPI(
    title="Unicorn Predictor API",
    description="Prédit le succès des startups (supervisé) et leur cluster (non supervisé).",
)


# 2. Chargement des modèles (les fichiers .pkl générés dans ton notebook)
BASE_DIR = Path(__file__).resolve().parent

def load_artifact(path: Path) -> Optional[object]:
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Impossible de charger {path}: {exc}")
        return None


model_supervised = load_artifact(BASE_DIR.parent / "models" / "unicorn_model.pkl")
model_clusters = load_artifact(BASE_DIR.parent / "models" / "unicorn_clusters.pkl")


# 3. Structures des données (Pydantic)
class SupervisedInput(BaseModel):
    category_code: str
    country_code: str
    funding_total_usd: float
    relationships: int
    total_funding_usd: float
    investor_count: int
    cluster_profile: int


class ClusterInput(BaseModel):
    funding_total_usd: float
    funding_rounds: float
    relationships: float
    total_funding_usd: float
    funding_rounds_count: float
    investor_count: float


# 4. Healthcheck simple
@app.get("/")
def home():
    return {
        "message": "API Unicorn Predictor est en ligne !",
        "supervised_loaded": model_supervised is not None,
        "clustering_loaded": model_clusters is not None,
    }


# 5. Route Supervisée : succès / probabilité
@app.post("/predict")
def predict(data: SupervisedInput):
    if model_supervised is None:
        return {"error": "Modèle supervisé non chargé. Vérifiez unicorn_model.pkl"}

    df_input = pd.DataFrame([data.dict()])
    prediction = model_supervised.predict(df_input)[0]
    probability = model_supervised.predict_proba(df_input)[0][1]

    return {
        "status": "success",
        "is_unicorn": int(prediction),
        "confidence_score": f"{round(probability * 100, 2)}%",
        "recommendation": "Investissement prioritaire" if probability > 0.7 else "À surveiller",
    }


# 6. Route Non Supervisée : cluster KMeans
@app.post("/cluster")
def cluster(data: ClusterInput):
    if model_clusters is None:
        return {"error": "Modèle de clustering non chargé. Vérifiez unicorn_clusters.pkl"}

    # Respecter l'ordre des 6 features attendues par le KMeans
    features = [
        "funding_total_usd",
        "funding_rounds",
        "relationships",
        "total_funding_usd",
        "funding_rounds_count",
        "investor_count",
    ]
    df_input = pd.DataFrame([data.dict()])[features]

    label = int(model_clusters.predict(df_input)[0])

    profile = {
        0: "Cluster 0 : faible probabilité (historique ~4%)",
        1: "Cluster 1 : profil intermédiaire",
        2: "Cluster 2 : cluster élite (historique ~54%)",
    }.get(label, f"Cluster {label}")

    return {
        "status": "success",
        "cluster_label": label,
        "cluster_profile": profile,
    }
