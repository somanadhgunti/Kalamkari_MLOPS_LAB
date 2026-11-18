"""FastAPI service for Kalamkari fabric drying-time predictions."""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Kalamkari Fabric Drying Time Prediction API",
    description="Predicts fabric drying duration using the trained model.",
)


class FabricInput(BaseModel):
    """Validated raw fabric data accepted by the prediction endpoint."""

    fabric_material: Literal["Cotton", "Silk"]
    fabric_length: float = Field(gt=0, le=20)
    dye_coats: int = Field(ge=1, le=6)
    humidity: float = Field(ge=0, le=100)
    temperature: float = Field(gt=0, le=100)
    airflow_speed: float = Field(gt=0, le=100)
    rack_occupancy: float = Field(ge=0, le=100)


@app.get("/")
def root() -> dict[str, str]:
    """Report that the API is running."""
    return {"message": "Kalamkari Fabric Drying Time Prediction API is running."}


@app.post("/predict")
def predict(fabric: FabricInput) -> dict[str, float | str]:
    """Predict drying duration for one raw fabric record."""
    input_data = pd.DataFrame([fabric.model_dump()])
    prediction = max(0.0, float(model.predict(input_data)[0]))
    return {
        "predicted_drying_time_hours": round(prediction, 2),
        "unit": "hours",
    }
