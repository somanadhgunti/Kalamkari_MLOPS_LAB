"""Prepare Kalamkari drying data for machine-learning experiments."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import FeatureEngineeringTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
FEATURE_STORE_DIR = PROJECT_ROOT / "feature_store"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET_COLUMN = "drying_duration"
ENGINEERED_FEATURES = ["moisture_load_index", "airflow_per_meter"]


def build_preprocessing_pipeline(
    numerical_columns: list[str], categorical_columns: list[str]
) -> Pipeline:
    """Build a reusable feature-engineering and preprocessing pipeline."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineeringTransformer()),
            ("preprocessing", column_transformer),
        ]
    )


def load_and_combine_datasets() -> pd.DataFrame:
    """Load both existing versions and label their source version."""
    normal_data = pd.read_csv(DATASETS_DIR / "fabric_v1.csv")
    monsoon_data = pd.read_csv(DATASETS_DIR / "fabric_v2_monsoon.csv")
    normal_data["dataset_version"] = "normal"
    monsoon_data["dataset_version"] = "monsoon"
    return pd.concat([normal_data, monsoon_data], ignore_index=True)


def main() -> None:
    """Create and save the processed feature store and fitted pipeline."""
    combined_data = load_and_combine_datasets()
    print(f"Original dataset shape: {combined_data.shape}")

    # Keep the target outside the input features to prevent target leakage.
    features = combined_data.drop(columns=[TARGET_COLUMN])
    target = combined_data[[TARGET_COLUMN]]

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

    pipeline = build_preprocessing_pipeline(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )
    processed_features = pipeline.fit_transform(features)
    feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()

    processed_data = pd.DataFrame(
        processed_features,
        columns=feature_names,
        index=combined_data.index,
    )

    FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    processed_data.to_csv(FEATURE_STORE_DIR / "fabric_features.csv", index=False)
    target.to_csv(FEATURE_STORE_DIR / "target.csv", index=False)
    joblib.dump(pipeline, MODELS_DIR / "preprocessor.pkl")

    reused_features = pd.read_csv(FEATURE_STORE_DIR / "fabric_features.csv")
    print(f"Processed feature shape: {reused_features.shape}")
    print(f"Engineered feature names: {ENGINEERED_FEATURES}")
    print("\nFirst five processed records:")
    print(reused_features.head().to_string(index=False))
    print("\nFeature store created:")
    print("  feature_store/fabric_features.csv")
    print("  feature_store/target.csv")
    print("  models/preprocessor.pkl")


if __name__ == "__main__":
    main()
