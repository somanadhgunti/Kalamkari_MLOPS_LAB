"""Tests for the saved Kalamkari drying-time prediction pipeline."""

from pathlib import Path

import joblib
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
REQUIRED_COLUMNS = [
    "fabric_material",
    "fabric_length",
    "dye_coats",
    "humidity",
    "temperature",
    "airflow_speed",
    "rack_occupancy",
]


def valid_input() -> pd.DataFrame:
    """Return one valid raw fabric record."""
    return pd.DataFrame(
        [
            {
                "fabric_material": "Cotton",
                "fabric_length": 10.0,
                "dye_coats": 3,
                "humidity": 60.0,
                "temperature": 32.0,
                "airflow_speed": 2.5,
                "rack_occupancy": 65.0,
            }
        ],
        columns=REQUIRED_COLUMNS,
    )


def test_saved_model_file_exists():
    assert MODEL_PATH.is_file()


def test_saved_model_loads_successfully():
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict")


def test_valid_input_produces_non_negative_numerical_prediction():
    model = joblib.load(MODEL_PATH)

    prediction = model.predict(valid_input())

    assert len(prediction) == 1
    assert isinstance(float(prediction[0]), float)
    assert prediction[0] >= 0


def test_missing_required_input_columns_are_rejected():
    model = joblib.load(MODEL_PATH)
    incomplete_input = valid_input().drop(columns=["humidity"])

    with pytest.raises((KeyError, ValueError)):
        model.predict(incomplete_input)
