
from pathlib import Path
import math

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "phishguard_model.joblib"

app = FastAPI(
    title="PhishGuard XAI API",
    description=(
        "REST API for phishing classification using a trained "
        "machine learning model. Submit extracted numeric features."
    ),
    version="1.0.0"
)

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model file not found: {MODEL_PATH}. Run src/train.py first."
    )

saved = joblib.load(MODEL_PATH)
model = saved["model"]
FEATURES = saved["features"]
MODEL_NAME = saved["best_model_name"]


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description=(
            "Numeric feature values matching the training feature names."
        )
    )


@app.get("/")
def home():
    return {
        "project": "PhishGuard XAI",
        "message": "Phishing classification API is running.",
        "model": MODEL_NAME,
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": MODEL_NAME,
        "feature_count": len(FEATURES)
    }


@app.get("/features")
def get_features():
    return {
        "feature_count": len(FEATURES),
        "required_features": FEATURES
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    supplied = request.features
    expected = set(FEATURES)
    received = set(supplied)

    missing = sorted(expected - received)
    unexpected = sorted(received - expected)

    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Input feature names do not match the model.",
                "missing_features": missing,
                "unexpected_features": unexpected
            }
        )

    for name, value in supplied.items():
        if not math.isfinite(value):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{name}' must be a finite number."
            )

    try:
        input_df = pd.DataFrame(
            [[supplied[name] for name in FEATURES]],
            columns=FEATURES
        )

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        probability_label_1 = (
            float(probabilities[classes.index(1)])
            if 1 in classes else None
        )

        return {
            "model": MODEL_NAME,
            "predicted_label": prediction,
            "probability_label_1": probability_label_1,
            "label_note": (
                "Interpret label 0/1 using the dataset's verified "
                "label mapping. This endpoint does not scan a URL."
            )
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}: {exc}"
        )
