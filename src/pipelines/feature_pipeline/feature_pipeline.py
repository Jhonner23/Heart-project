"""Feature pipeline: validates, cleans and transforms raw heart-disease data.

Reads the raw CSV, validates it (types, ranges, null rates, valid
categories, and a domain-specific cross-field integrity rule), creates a
reproducible train/test split, fits a preprocessing pipeline using ONLY the
training rows, then applies that fitted preprocessor to the full validated
dataset and persists the transformed feature matrix (with a ``split``
column and the target) plus the fitted preprocessor.

Validation policy (see ``validate_raw_data``):
- Dataset-level thresholds (max null % per column, max fraction of rows
  failing schema checks) are a hard gate — if breached, a
  ``DataValidationError`` is raised and NOTHING is persisted.
- Individual rows failing an element-wise check (out-of-range value,
  invalid category, physiologically implausible reading) are dropped and
  logged rather than failing the whole run, since this dataset has
  deliberately injected row-level corruption (see
  data/01_raw/datos_corazon_Info.txt).

Not applicable to this dataset: date-format rules (no date/datetime
columns exist) and key-field uniqueness (there is no natural ID column —
the closest equivalent, exact full-row duplicates, is handled as
structural cleanup in ``load_raw_data``, not as a validation failure).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandera.pandas as pa
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
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
SPLIT_COL: str = "split"

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

# --- Validation thresholds (see docstring above for the two-tier policy) ---
MAX_NULL_PCT: float = 0.30
MAX_INVALID_ROW_FRACTION: float = 0.10
HR_MAX_TOLERANCE: float = 20.0  # beats/min over the classic 220-age estimate

NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "age": (0, 120),
    "rest_bp": (50, 250),
    "chol": (0, 700),
    "max_hr": (50, 250),
    "old_peak": (0, 10),
}

VALID_CATEGORIES: dict[str, list[str]] = {
    "sex": ["Male", "Female"],
    "chest_pain": ["typical", "asymptomatic", "nonanginal", "nontypical"],
    "fbs": ["0", "1"],
    "rest_ecg": ["normal", "left ventricular hypertrophy", "ST-T wave abnormality"],
    "slope": ["1", "2", "3"],
    "ca": ["0", "1", "2", "3"],
    "thal": ["normal", "fixed", "reversable"],
    "exang": ["0", "1"],
}


RAW_SCHEMA = pa.DataFrameSchema(
    {
        **{
            col: pa.Column(float, pa.Check.in_range(lo, hi), nullable=True)
            for col, (lo, hi) in NUMERIC_RANGES.items()
        },
        **{
            col: pa.Column(object, pa.Check.isin(cats), nullable=True)
            for col, cats in VALID_CATEGORIES.items()
        },
        TARGET: pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
    },
    strict=False,
    coerce=False,
)


class DataValidationError(ValueError):
    """Raised when raw data fails validation badly enough to block persistence."""


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
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_COLS),
            ("cat", categorical_transformer, CAT_COLS),
        ]
    )


def _normalize_categorical(value: object) -> object:
    """Normalize a categorical value to a consistent string representation.

    Fixes two real data-format issues found in the raw CSV: (1) some
    boolean-like columns mix float (0.0/1.0) and string ("0"/"1")
    representations across rows, and (2) some category strings have
    trailing whitespace (e.g. "left ventricular hypertrophy "). Leaves
    genuinely non-numeric text (including garbage values) untouched so
    validation can flag it.
    """
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    try:
        as_float = float(text)
    except ValueError:
        return text
    return str(int(as_float)) if as_float.is_integer() else text


def load_raw_data(raw_path: Path) -> pd.DataFrame:
    """Load the raw CSV and apply the minimal, deterministic type fixes.

    Structural cleanup only: duplicates, numeric coercion, categorical
    normalization, and dropping rows with no usable target (a row without a
    valid label can't be used for supervised learning regardless of any
    later validation rule). Rejecting bad *feature* values is
    ``validate_raw_data``'s job, not this function's.
    """
    df = pd.read_csv(raw_path)
    df = df.drop_duplicates().reset_index(drop=True)

    missing_cols = set(NUM_COLS + CAT_COLS + [TARGET]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Raw data is missing expected columns: {sorted(missing_cols)}")

    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CAT_COLS:
        df[col] = df[col].map(_normalize_categorical)

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    n_before_target_filter = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    if len(df) < n_before_target_filter:
        logger.warning(
            "Dropped %d rows with missing/invalid target (%s)",
            n_before_target_filter - len(df),
            TARGET,
        )
    df[TARGET] = df[TARGET].astype(int)

    return df


def _check_heart_rate_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-field integrity rule specific to this problem: ``max_hr``
    (measured maximum heart rate) shouldn't substantially exceed the
    well-known physiological estimate of maximum heart rate, ``220 - age``,
    even allowing generous tolerance. Rows violating this are logged and
    dropped as physiologically implausible records.
    """
    estimated_max = 220 - df["age"] + HR_MAX_TOLERANCE
    violates = (df["max_hr"] > estimated_max).fillna(False)
    if violates.any():
        logger.warning(
            "Dropping %d rows failing the integrity rule max_hr <= 220 - age + %.0f",
            int(violates.sum()),
            HR_MAX_TOLERANCE,
        )
        df = df.loc[~violates].reset_index(drop=True)
    return df


def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the cleaned data before it becomes a feature table.

    Raises ``DataValidationError`` (blocking any persistence) if:
    - any column's null rate exceeds ``MAX_NULL_PCT``, or
    - the fraction of rows failing the schema (type/range/category checks)
      exceeds ``MAX_INVALID_ROW_FRACTION``.

    Otherwise, rows that fail an individual check are dropped and logged,
    and the remaining valid rows are returned.
    """
    null_pct = df[NUM_COLS + CAT_COLS].isna().mean()
    breaches = null_pct[null_pct > MAX_NULL_PCT]
    if not breaches.empty:
        raise DataValidationError(
            f"Column(s) exceed the {MAX_NULL_PCT:.0%} max null rate: {breaches.round(3).to_dict()}"
        )

    try:
        RAW_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        failure_cases = exc.failure_cases
        invalid_idx = set(failure_cases["index"].dropna().astype(int))
        invalid_fraction = len(invalid_idx) / len(df)

        if invalid_fraction > MAX_INVALID_ROW_FRACTION:
            raise DataValidationError(
                f"{invalid_fraction:.1%} of rows failed schema validation "
                f"(max allowed: {MAX_INVALID_ROW_FRACTION:.0%}). Sample failures:\n"
                f"{failure_cases[['column', 'check', 'failure_case']].head(10)}"
            ) from exc

        logger.warning(
            "Dropping %d/%d rows (%.1f%%) that failed schema validation",
            len(invalid_idx),
            len(df),
            invalid_fraction * 100,
        )
        return df.drop(index=list(invalid_idx)).reset_index(drop=True)
    else:
        return df


ROOT_DIR: Path = Path(__file__).resolve().parents[3]
DEFAULT_RAW_PATH: Path = ROOT_DIR / "data" / "01_raw" / "corazon.csv"
DEFAULT_FEATURES_PATH: Path = ROOT_DIR / "data" / "04_feature" / "corazon_features.parquet"
DEFAULT_PIPELINE_PATH: Path = ROOT_DIR / "models" / "feature_pipeline.pkl"


def run_feature_pipeline(
    raw_path: Path = DEFAULT_RAW_PATH,
    features_path: Path = DEFAULT_FEATURES_PATH,
    pipeline_path: Path = DEFAULT_PIPELINE_PATH,
) -> pd.DataFrame:
    """Run the full feature pipeline: load -> validate -> split -> fit(train)
    -> transform(all) -> persist.

    Nothing is written to ``features_path``/``pipeline_path`` if
    ``validate_raw_data`` raises ``DataValidationError``.
    """
    logger.info("Loading raw data from %s", raw_path)
    df = load_raw_data(raw_path)

    df = validate_raw_data(df)
    df = _check_heart_rate_integrity(df)

    features = df.drop(columns=[TARGET])
    target = df[TARGET]

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    split = pd.Series(SPLIT_COL, index=df.index, dtype=object)
    split.loc[train_idx] = "train"
    split.loc[test_idx] = "test"

    preprocessor = build_preprocessor()
    logger.info(
        "Fitting preprocessor on %d training rows (of %d total)",
        len(train_idx),
        len(df),
    )
    preprocessor.fit(features.loc[train_idx])

    # transform (not fit_transform) on the FULL dataset: reuses train-only
    # statistics, so no test-set information leaks into imputation/scaling.
    transformed = preprocessor.transform(features)
    feature_names = preprocessor.get_feature_names_out()

    features_df = pd.DataFrame(transformed, columns=feature_names, index=df.index)
    features_df[TARGET] = target
    features_df[SPLIT_COL] = split
    features_df = features_df.reset_index(drop=True)

    features_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)

    features_df.to_parquet(features_path, index=False)
    joblib.dump(preprocessor, pipeline_path)

    logger.info("Saved %d transformed features to %s", len(feature_names), features_path)
    logger.info("Saved fitted preprocessor to %s", pipeline_path)
    return features_df


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
