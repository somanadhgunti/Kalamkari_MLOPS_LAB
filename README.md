# Estimating Drying Time for Kalamkari Fabric Production using MLOps

This MLOps laboratory project predicts the drying time of Kalamkari fabric batches in hours. It uses reproducible synthetic data, preprocessing and feature engineering, two regression models, MLflow tracking and registration, a FastAPI prediction service, pytest tests, and GitHub Actions.

## Problem statement

Kalamkari fabric production near Machilipatnam requires fabric batches to dry after dyeing. Drying time is affected by humidity, temperature, airflow, fabric length, dye coats, and rack occupancy. This project estimates drying duration so that a batch can be planned more effectively.

## Features

- Generates normal-weather and monsoon synthetic datasets.
- Builds a reusable preprocessing and feature-engineering pipeline.
- Trains Ridge Regression and Random Forest Regression models.
- Evaluates models with MAE, RMSE, and R².
- Tracks runs and registers the selected model with MLflow.
- Serves predictions through FastAPI.
- Tests the saved model with pytest.
- Runs tests automatically on pushes and pull requests with GitHub Actions.

## Technologies

Python, Pandas, NumPy, Scikit-learn, Joblib, MLflow, FastAPI, Uvicorn, Pytest, and GitHub Actions.

## Project structure

```text
mlops_lab/
├── app.py
├── dataset_versions.txt
├── requirements.txt
├── README.md
├── datasets/
│   ├── fabric_v1.csv
│   └── fabric_v2_monsoon.csv
├── feature_store/
│   ├── fabric_features.csv
│   └── target.csv
├── models/
│   ├── best_model.pkl
│   ├── model_comparison.csv
│   └── preprocessor.pkl
├── report/
│   └── Implementation_Report.pdf
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py
│   ├── generate_data.py
│   ├── preprocess.py
│   └── train.py
├── tests/
│   └── test_model.py
└── .github/workflows/
    └── main.yml
```

## Dataset details

The datasets are synthetic and contain 300 records each:

| Version | Conditions | Humidity | Records | Average drying duration |
|---|---|---:|---:|---:|
| `fabric_v1.csv` | Normal weather | 40.13%–69.96% | 300 | 9.6877 hours |
| `fabric_v2_monsoon.csv` | Monsoon weather | 75.01%–94.98% | 300 | 11.7426 hours |

The combined feature-store preparation data contains 600 records. Higher monsoon humidity increases the estimated drying duration.

## Feature engineering

The preprocessing pipeline creates:

```text
moisture_load_index = fabric_length * dye_coats * (humidity / 100)
airflow_per_meter = airflow_speed / fabric_length
```

Numerical values are mean-imputed and standardized. Categorical values are filled with the most frequent value and one-hot encoded.

## Model evaluation

The data is split into 80% training data and 20% test data using `random_state=42`.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge Regression | 0.199617 | 0.254202 | 0.984733 |
| Random Forest Regressor | 0.510354 | 0.626854 | 0.907163 |

Ridge Regression was selected because it achieved the lowest MAE and RMSE.

## MLflow registration

- Registered model: `KalamkariDryingTimeModel`
- Registered version: `2`
- Local tracking database: `mlflow.db`

## Installation

```powershell
cd C:\Users\Somu\Desktop\mlops_lab
python -m pip install -r requirements.txt
```

## Commands

Generate the synthetic datasets:

```powershell
python src\generate_data.py
```

Prepare the feature store:

```powershell
python -m src.preprocess
```

Train and register the model:

```powershell
python -m src.train
```

Run tests:

```powershell
python -m pytest
```

Start the API:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

## API example

Request:

```json
{
  "fabric_material": "Cotton",
  "fabric_length": 10.0,
  "dye_coats": 3,
  "humidity": 60.0,
  "temperature": 32.0,
  "airflow_speed": 2.5,
  "rack_occupancy": 65.0
}
```

Example response from the existing model:

```json
{
  "predicted_drying_time_hours": 9.34,
  "unit": "hours"
}
```

## GitHub Actions

The workflow in `.github/workflows/main.yml` runs on every push and pull request. It uses Python 3.11, installs `requirements.txt`, and executes the pytest suite. No deployment or cloud service is claimed by this project.

## Limitations and future enhancements

This laboratory implementation uses synthetic data. Real production data would be required for real-world deployment. Future work could collect real Kalamkari production data, deploy the API to a cloud platform, and add monitoring with scheduled model retraining.
