"""Reusable feature engineering transformer for the preprocessing pipeline."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


ENGINEERED_FEATURES = ["moisture_load_index", "airflow_per_meter"]


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add the two domain features needed by the drying-time model."""
    data = data.copy()
    data["moisture_load_index"] = (
        data["fabric_length"] * data["dye_coats"] * (data["humidity"] / 100)
    )
    data["airflow_per_meter"] = data["airflow_speed"] / data["fabric_length"]
    return data


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer that adds the domain features."""

    def fit(self, data: pd.DataFrame, target: pd.Series | None = None):
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return add_engineered_features(data)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        if input_features is None:
            return ENGINEERED_FEATURES
        return list(input_features) + ENGINEERED_FEATURES


class DatasetVersionTransformer(BaseEstimator, TransformerMixin):
    """Add the normal-weather version when raw prediction input omits it."""

    def fit(self, data: pd.DataFrame, target: pd.Series | None = None):
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        if "dataset_version" not in data.columns:
            data["dataset_version"] = "normal"
        return data

    def get_feature_names_out(self, input_features=None) -> list[str]:
        if input_features is None:
            return ["dataset_version"]
        if "dataset_version" in input_features:
            return list(input_features)
        return list(input_features) + ["dataset_version"]
