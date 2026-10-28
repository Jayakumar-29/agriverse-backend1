
from pathlib import Path
import json
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR

with open(BASE_DIR / "feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

MODELS = {}
for path in MODEL_DIR.glob("*.joblib"):
    name = path.stem.replace("_", " ").title()
    MODELS[name] = joblib.load(path)

app = FastAPI(
    title="AgriVerse Crop Yield API",
    version="1.0.0"
)

# For initial development only. Restrict this to your deployed
# frontend's exact origin before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class FarmInput(BaseModel):
    features: Dict[str, float]

@app.get("/")
def home():
    return {
        "project": "AgriVerse",
        "status": "running",
        "models": list(MODELS.keys())
    }

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(MODELS)}

@app.post("/predict")
def predict(request: FarmInput):
    missing = [c for c in FEATURE_COLUMNS
               if c not in request.features]
    extra = [c for c in request.features
             if c not in FEATURE_COLUMNS]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing,
                    "unexpected_features": extra}
        )

    try:
        row = pd.DataFrame(
            [[request.features[c] for c in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS
        )

        predictions = {
            name: float(model.predict(row)[0])
            for name, model in MODELS.items()
        }

        return {
            "predictions_kg_per_hectare": predictions,
            "data_note": "Demonstration models trained on synthetic data."
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs."
        ) from exc
