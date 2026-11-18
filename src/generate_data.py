"""Generate reproducible synthetic Kalamkari fabric drying datasets."""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
RECORD_COUNT = 300
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"


def calculate_drying_duration(
    fabric_material: pd.Series,
    fabric_length: pd.Series,
    dye_coats: pd.Series,
    humidity: pd.Series,
    temperature: pd.Series,
    airflow_speed: pd.Series,
    rack_occupancy: pd.Series,
    rng: np.random.Generator,
) -> np.ndarray:
    """Calculate a simple, realistic drying-duration formula in hours."""
    material_effect = np.where(fabric_material == "Silk", 0.8, 0.0)
    noise = rng.normal(loc=0.0, scale=0.25, size=len(fabric_length))

    duration = (
        5.0
        + 0.18 * fabric_length
        + 0.65 * dye_coats
        + 0.07 * (humidity - 40)
        - 0.12 * (temperature - 25)
        - 0.35 * airflow_speed
        + 0.02 * (rack_occupancy - 20)
        + material_effect
        + noise
    )

    # Keep the target positive and round it for easy reading.
    return np.round(np.maximum(duration, 0.5), 2)


def create_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create normal-weather and monsoon-weather datasets."""
    rng = np.random.default_rng(RANDOM_SEED)

    # Generate shared fabric-batch properties once for both versions.
    fabric_material = rng.choice(["Cotton", "Silk"], size=RECORD_COUNT)
    fabric_length = np.round(rng.uniform(2, 20, size=RECORD_COUNT), 2)
    dye_coats = rng.integers(1, 7, size=RECORD_COUNT)
    temperature = np.round(rng.uniform(25, 40, size=RECORD_COUNT), 2)
    airflow_speed = np.round(rng.uniform(0.5, 5, size=RECORD_COUNT), 2)
    rack_occupancy = np.round(rng.uniform(20, 100, size=RECORD_COUNT), 2)
    normal_humidity = np.round(rng.uniform(40, 70, size=RECORD_COUNT), 2)
    monsoon_humidity = np.round(rng.uniform(75, 95, size=RECORD_COUNT), 2)

    common_properties = {
        "fabric_material": fabric_material,
        "fabric_length": fabric_length,
        "dye_coats": dye_coats,
        "temperature": temperature,
        "airflow_speed": airflow_speed,
        "rack_occupancy": rack_occupancy,
    }

    normal_data = pd.DataFrame(
        {
            **common_properties,
            "humidity": normal_humidity,
        }
    )
    normal_data = normal_data[
        [
            "fabric_material",
            "fabric_length",
            "dye_coats",
            "humidity",
            "temperature",
            "airflow_speed",
            "rack_occupancy",
        ]
    ]
    normal_data["drying_duration"] = calculate_drying_duration(
        normal_data["fabric_material"],
        normal_data["fabric_length"],
        normal_data["dye_coats"],
        normal_data["humidity"],
        normal_data["temperature"],
        normal_data["airflow_speed"],
        normal_data["rack_occupancy"],
        rng,
    )

    monsoon_data = pd.DataFrame(
        {
            **common_properties,
            "humidity": monsoon_humidity,
        }
    )
    monsoon_data = monsoon_data[
        [
            "fabric_material",
            "fabric_length",
            "dye_coats",
            "humidity",
            "temperature",
            "airflow_speed",
            "rack_occupancy",
        ]
    ]
    monsoon_data["drying_duration"] = calculate_drying_duration(
        monsoon_data["fabric_material"],
        monsoon_data["fabric_length"],
        monsoon_data["dye_coats"],
        monsoon_data["humidity"],
        monsoon_data["temperature"],
        monsoon_data["airflow_speed"],
        monsoon_data["rack_occupancy"],
        rng,
    )

    return normal_data, monsoon_data


def print_summary(normal_data: pd.DataFrame, monsoon_data: pd.DataFrame) -> None:
    """Display the requested dataset previews and comparison."""
    print("Normal-weather dataset (first 5 rows):")
    print(normal_data.head().to_string(index=False))
    print("\nMonsoon-weather dataset (first 5 rows):")
    print(monsoon_data.head().to_string(index=False))

    print(f"\nNormal dataset shape: {normal_data.shape}")
    print(f"Monsoon dataset shape: {monsoon_data.shape}")
    print(f"Normal average drying duration: {normal_data['drying_duration'].mean():.2f} hours")
    print(f"Monsoon average drying duration: {monsoon_data['drying_duration'].mean():.2f} hours")
    print(
        "Normal humidity range: "
        f"{normal_data['humidity'].min():.2f}% to {normal_data['humidity'].max():.2f}%"
    )
    print(
        "Monsoon humidity range: "
        f"{monsoon_data['humidity'].min():.2f}% to {monsoon_data['humidity'].max():.2f}%"
    )
    print("\nAverage drying-duration comparison:")
    print(
        f"Normal: {normal_data['drying_duration'].mean():.2f} hours"
        f"  ->  Monsoon: {monsoon_data['drying_duration'].mean():.2f} hours"
    )


def main() -> None:
    """Generate and save both CSV versions."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    normal_data, monsoon_data = create_datasets()

    normal_data.to_csv(DATASETS_DIR / "fabric_v1.csv", index=False)
    monsoon_data.to_csv(DATASETS_DIR / "fabric_v2_monsoon.csv", index=False)
    print_summary(normal_data, monsoon_data)


if __name__ == "__main__":
    main()
