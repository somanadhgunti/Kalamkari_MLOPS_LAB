"""Train, evaluate, and register Kalamkari drying-time models."""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.feature_engineering import DatasetVersionTransformer
from src.preprocess import (
    ENGINEERED_FEATURES,
    TARGET_COLUMN,
    build_preprocessing_pipeline,
    load_and_combine_datasets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_STORE_DIR = PROJECT_ROOT / "feature_store"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
EXPERIMENT_NAME = "Kalamkari Drying Time"
REGISTERED_MODEL_NAME = "KalamkariDryingTimeModel"


def build_raw_preprocessor() -> Pipeline:
    """Create the same feature engineering and preprocessing used previously."""
    numerical_columns = [
        "fabric_length",
        "dye_coats",
        "humidity",
        "temperature",
        "airflow_speed",
        "rack_occupancy",
        *ENGINEERED_FEATURES,
    ]
    categorical_columns = ["fabric_material", "dataset_version"]
    return build_preprocessing_pipeline(numerical_columns, categorical_columns)


def evaluate_model(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict:
    """Fit a model and return its test metrics."""
    model.fit(features["train"], target["train"])
    predictions = model.predict(features["test"])
    return {
        "MAE": mean_absolute_error(target["test"], predictions),
        "RMSE": np.sqrt(mean_squared_error(target["test"], predictions)),
        "R2": r2_score(target["test"], predictions),
    }


def configure_mlflow() -> None:
    """Configure local SQLite-backed MLflow tracking and model registry."""
    tracking_uri = f"sqlite:///{MLFLOW_DB.resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def main() -> None:
    """Train both models, save results, and register the best model."""
    # Load the requested feature-store files.
    feature_store_features = pd.read_csv(
        FEATURE_STORE_DIR / "fabric_features.csv"
    )
    target_data = pd.read_csv(FEATURE_STORE_DIR / "target.csv")
    target_values = target_data[TARGET_COLUMN]

    if len(feature_store_features) != len(target_values):
        raise ValueError("Feature and target files contain different row counts.")

    # Use the same row indices to split raw data and keep the target separate.
    raw_data = load_and_combine_datasets()
    raw_features = raw_data.drop(columns=[TARGET_COLUMN])
    indices = np.arange(len(feature_store_features))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42
    )
    split_features = {
        "train": raw_features.iloc[train_indices],
        "test": raw_features.iloc[test_indices],
    }
    split_target = {
        "train": target_values.iloc[train_indices],
        "test": target_values.iloc[test_indices],
    }

    models = {
        "Ridge Regression": Ridge(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
    }
    results = []
    trained_models = {}

    configure_mlflow()
    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("dataset_version", DatasetVersionTransformer()),
                ("preprocessor", clone(build_raw_preprocessor())),
                ("model", estimator),
            ]
        )
        with mlflow.start_run(run_name=model_name) as run:
            metrics = evaluate_model(pipeline, split_features, split_target)
            mlflow.log_param("model_name", model_name)
            for parameter, value in estimator.get_params().items():
                if parameter in {"alpha", "n_estimators", "random_state"}:
                    mlflow.log_param(parameter, value)
            mlflow.log_metrics(
                {
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "R2": metrics["R2"],
                }
            )
            mlflow.set_tag("dataset", "Kalamkari fabric feature store")
            results.append({"model": model_name, **metrics})
            trained_models[model_name] = pipeline
            print(f"MLflow run logged: {model_name} (run_id={run.info.run_id})")

    comparison = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    comparison.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    best_model_name = comparison.iloc[0]["model"]
    best_pipeline = trained_models[best_model_name]

    best_run_name = f"{best_model_name} - registered"
    with mlflow.start_run(run_name=best_run_name) as run:
        best_metrics = comparison.iloc[0]
        mlflow.log_param("model_name", best_model_name)
        mlflow.log_metrics(
            {
                "MAE": best_metrics["MAE"],
                "RMSE": best_metrics["RMSE"],
                "R2": best_metrics["R2"],
            }
        )
        model_info = mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="model",
            serialization_format="cloudpickle",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        registered_run_id = run.info.run_id

    joblib.dump(best_pipeline, MODELS_DIR / "best_model.pkl")

    client = MlflowClient()
    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )
    matching_versions = [
        version for version in versions if version.run_id == registered_run_id
    ]
    if not matching_versions:
        raise RuntimeError("The registered MLflow model version could not be found.")
    registered_version = max(matching_versions, key=lambda version: int(version.version))

    print("\nModel comparison:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nBest model: {best_model_name}")
    print(
        "MLflow registered model: "
        f"{REGISTERED_MODEL_NAME}, version {registered_version.version}"
    )
    print("Saved complete prediction pipeline: models/best_model.pkl")
    print(f"MLflow model URI: {model_info.model_uri}")


if __name__ == "__main__":
    main()
