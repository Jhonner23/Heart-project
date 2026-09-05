"""Feature pipeline: transforms raw heart-disease data into model-ready features.

Reads the raw CSV, fits a preprocessing pipeline (imputation + scaling for
numeric columns, imputation + one-hot encoding for categorical columns) and
persists both the transformed dataset and the fitted preprocessor so that
``training_pipeline.py`` and ``inference_pipeline.py`` can reuse the exact
same transformations.

Data-quality validation (Pandera schema, null-rate checks, etc.) is added on
top of this script in the "Data Validation & Data Integrity" task; this
version focuses on the autonomous feature transformation itself.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Column groups for the "corazon" dataset (see data/01_raw/datos_corazon_Info.txt)
NUM_COLS: list[str] = ["age", "rest_bp", "chol", "max_hr", "old_peak"]
CAT_COLS: list[str] = [
    "sex",
    "chest_pain",
    "fbs",
    "rest_ecg",
    "slope",
    "ca",
    "thal",
    "exang",
]
TARGET: str = "disease"

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RAW_PATH = ROOT_DIR / "data" / "01_raw" / "corazon.csv"
DEFAULT_FEATURES_PATH = ROOT_DIR / "data" / "04_feature" / "corazon_features.parquet"
DEFAULT_PIPELINE_PATH = ROOT_DIR / "models" / "feature_pipeline.pkl"


def build_preprocessor() -> ColumnTransformer:
    """Build the (unfitted) numeric + categorical preprocessing pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_COLS),
            ("cat", categorical_transformer, CAT_COLS),
        ]
    )


def load_raw_data(raw_path: Path) -> pd.DataFrame:
    """Load the raw CSV and apply the minimal, deterministic type fixes.

    Only structural cleanup lives here (duplicates, numeric coercion). Data
    *validation* (rejecting bad data) belongs to the validation task.
    """
    df = pd.read_csv(raw_path)
    df = df.drop_duplicates().reset_index(drop=True)

    missing_cols = set(NUM_COLS + CAT_COLS + [TARGET]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Raw data is missing expected columns: {sorted(missing_cols)}")

    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CAT_COLS:
        df[col] = df[col].astype(object)

    return df


def run_feature_pipeline(
    raw_path: Path = DEFAULT_RAW_PATH,
    features_path: Path = DEFAULT_FEATURES_PATH,
    pipeline_path: Path = DEFAULT_PIPELINE_PATH,
) -> pd.DataFrame:
    """Run the full feature pipeline: load -> fit preprocessor -> persist.

    Returns the (untransformed, but cleaned) DataFrame that was persisted,
    which is what downstream pipelines read back from ``features_path``.
    """
    logger.info("Loading raw data from %s", raw_path)
    df = load_raw_data(raw_path)

    features = df.drop(columns=[TARGET])
    preprocessor = build_preprocessor()
    logger.info("Fitting preprocessor on %d rows", len(features))
    preprocessor.fit(features)

    features_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(features_path, index=False)
    joblib.dump(preprocessor, pipeline_path)

    logger.info("Saved features to %s", features_path)
    logger.info("Saved fitted preprocessor to %s", pipeline_path)
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the feature pipeline.")
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--pipeline-path", type=Path, default=DEFAULT_PIPELINE_PATH)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    run_feature_pipeline(
        raw_path=args.raw_path,
        features_path=args.features_path,
        pipeline_path=args.pipeline_path,
    )


if __name__ == "__main__":
    main()
