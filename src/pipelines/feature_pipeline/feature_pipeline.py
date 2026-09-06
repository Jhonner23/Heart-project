"""Feature pipeline: validates, cleans and transforms raw heart-disease data.

Reads the raw CSV, validates it (types, ranges, null rates, valid
categories, and a domain-specific cross-field integrity rule), creates a
reproducible train/test split, fits a preprocessing pipeline using ONLY the
training rows, then applies that fitted preprocessor to the full validated
dataset and persists the transformed feature matrix (with a ``split``
column and the target) plus the fitted preprocessor.

Data-quality architecture (two separate steps, deliberately not merged):
- ``clean_invalid_rows``: a data-CLEANING step (like the duplicate/target
  cleanup in ``load_raw_data``). Rows with known, recoverable corruption
  (out-of-range values, invalid categories, a physiologically implausible
  heart-rate reading) are dropped and logged - this dataset has
  deliberately injected row-level corruption (see
  data/01_raw/datos_corazon_Info.txt). If too large a fraction of rows
  need cleaning, the dataset itself is untrustworthy and this raises
  ``DataValidationError``.
- ``validate_raw_data``: the actual VALIDATION gate, run only on already
  -cleaned data. Every failure path here is fatal - a clear error, and
  NOTHING is persisted. No silent recovery happens inside this function.

Not applicable to this dataset: date-format rules (no date/datetime
columns exist) and key-field uniqueness (there is no natural ID column -
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

MAX_NULL_PCT: float = 0.30
MAX_INVALID_ROW_FRACTION: float = 0.10
HR_MAX_TOLERANCE: float = 20.0

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

ROOT_DIR: Path = Path(__file__).resolve().parents[3]
DEFAULT_RAW_PATH: Path = ROOT_DIR / "data" / "01_raw" / "corazon.csv"
DEFAULT_FEATURES_PATH: Path = ROOT_DIR / "data" / "04_feature" / "corazon_features.parquet"
DEFAULT_PIPELINE_PATH: Path = ROOT_DIR / "models" / "feature_pipeline.pkl"


class DataValidationError(ValueError):
    """Raised when raw data fails validation badly enough to block persistence."""


def build_preprocessor() -> ColumnTransformer:
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


def normalize_categorical(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    try:
        as_float = float(text)
    except ValueError:
        return text
    return str(int(as_float)) if as_float.is_integer() else text


def load_raw_data(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df = df.drop_duplicates().reset_index(drop=True)

    missing_cols = set(NUM_COLS + CAT_COLS + [TARGET]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Raw data is missing expected columns: {sorted(missing_cols)}")

    for col in NUM_COLS:
        # .astype(float) after to_numeric: without it, an all-integer column
        # with no nulls stays int64, which doesn't match RAW_SCHEMA's float
        # columns regardless of pandas version.
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in CAT_COLS:
        # .astype(object) pins the dtype explicitly: some pandas versions
        # infer a string extension dtype from .map() instead of plain
        # object, which RAW_SCHEMA's object columns don't match.
        df[col] = df[col].map(normalize_categorical).astype(object)

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
    """Drop rows violating the max_hr <= 220 - age (+ tolerance) rule.

    This is a cross-field integrity check used by ``clean_invalid_rows``,
    not a standalone validation step.
    """
    estimated_max = 220 - df["age"] + HR_MAX_TOLERANCE
    violates = (df["max_hr"] > estimated_max).fillna(False)
    if violates.any():
        logger.warning(
            "Cleaning %d rows violating max_hr <= 220 - age + %.0f",
            int(violates.sum()),
            HR_MAX_TOLERANCE,
        )
        df = df.loc[~violates].reset_index(drop=True)
    return df


def clean_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Data-cleaning step: remove rows with known, recoverable corruption.

    This dataset has deliberately injected row-level corruption (see
    data/01_raw/datos_corazon_Info.txt): out-of-range values, invalid
    categories, and physiologically implausible heart-rate readings. Rows
    with this kind of corruption are dropped and logged here - exactly like
    the duplicate-row and invalid-target cleanup already performed in
    ``load_raw_data`` - rather than treated as a validation failure.

    This is NOT the validation gate: if the fraction of rows needing this
    cleanup is itself too large to trust the dataset, that DOES raise
    ``DataValidationError`` (see ``MAX_INVALID_ROW_FRACTION``), because at
    that point no amount of row-dropping makes the data trustworthy.
    """
    try:
        RAW_SCHEMA.validate(df, lazy=True)
        cleaned = df
    except pa.errors.SchemaErrors as exc:
        failure_cases = exc.failure_cases
        invalid_idx = set(failure_cases["index"].dropna().astype(int))
        invalid_fraction = len(invalid_idx) / len(df)

        if invalid_fraction > MAX_INVALID_ROW_FRACTION:
            raise DataValidationError(
                f"{invalid_fraction:.1%} of rows failed schema checks "
                f"(max allowed: {MAX_INVALID_ROW_FRACTION:.0%}). Sample failures:\n"
                f"{failure_cases[['column', 'check', 'failure_case']].head(10)}"
            ) from exc

        logger.warning(
            "Cleaning %d/%d rows (%.1f%%) with schema-invalid values",
            len(invalid_idx),
            len(df),
            invalid_fraction * 100,
        )
        cleaned = df.drop(index=list(invalid_idx)).reset_index(drop=True)

    return _check_heart_rate_integrity(cleaned)


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validation gate: run on already-cleaned data (see ``clean_invalid_rows``).

    Raises ``DataValidationError`` - with no recovery, blocking any
    persistence - if:
    - any column's null rate exceeds ``MAX_NULL_PCT``, or
    - the data still fails the schema after cleaning (should not normally
      happen, but this is enforced strictly as a final gate rather than
      silently tolerated).

    Every failure path here is fatal by design: unlike ``clean_invalid_rows``,
    this function never drops rows and never returns partial data.
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
        raise DataValidationError(
            "Data still fails schema validation after cleaning "
            f"(this should not happen):\n"
            f"{exc.failure_cases[['column', 'check', 'failure_case']].head(10)}"
        ) from exc


def run_feature_pipeline(
    raw_path: Path = DEFAULT_RAW_PATH,
    features_path: Path = DEFAULT_FEATURES_PATH,
    pipeline_path: Path = DEFAULT_PIPELINE_PATH,
) -> pd.DataFrame:
    """Run the full feature pipeline: load -> clean -> validate -> split ->
    fit(train) -> transform(all) -> persist.

    Nothing is written to ``features_path``/``pipeline_path`` if
    ``validate_raw_data`` raises ``DataValidationError``.
    """
    logger.info("Loading raw data from %s", raw_path)
    df = load_raw_data(raw_path)

    df = clean_invalid_rows(df)
    validate_raw_data(df)

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
