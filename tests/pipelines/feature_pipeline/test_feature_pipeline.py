"""Unit tests for the feature pipeline (transformation + data validation)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.pipelines.feature_pipeline.feature_pipeline import (
    CAT_COLS,
    MAX_INVALID_ROW_FRACTION,
    MAX_NULL_PCT,
    NUM_COLS,
    SPLIT_COL,
    TARGET,
    DataValidationError,
    build_preprocessor,
    clean_invalid_rows,
    load_raw_data,
    run_feature_pipeline,
    validate_raw_data,
)

EXPECTED_ROWS_AFTER_DEDUP = 4
EXPECTED_ROWS_AFTER_TARGET_FILTER = 2


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Small synthetic dataset shaped like corazon.csv, including edge cases:
    a duplicated row, a missing numeric value and a missing categorical value.
    """
    data = {
        "age": [63, 67, 67, 45, np.nan],
        "sex": ["Male", "Female", "Female", "Male", "Male"],
        "chest_pain": [
            "typical",
            "asymptomatic",
            "asymptomatic",
            "nontypical",
            "typical",
        ],
        "rest_bp": [145, 160, 160, 130, 120],
        "chol": [233, 286, 286, 250, 210],
        "fbs": [1, 0, 0, 0, 1],
        "rest_ecg": [
            "left ventricular hypertrophy",
            "normal",
            "normal",
            "normal",
            "normal",
        ],
        "max_hr": [150, 108, 108, 170, None],
        "exang": [0, 1, 1, 0, 0],
        "old_peak": [2.3, 1.5, 1.5, 0.0, 1.2],
        "slope": [3, 2, 2, 1, 1],
        "ca": [0.0, 3.0, 3.0, None, 1.0],
        "thal": ["fixed", "normal", "normal", "normal", None],
        "disease": [0, 1, 1, 0, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_raw_csv(tmp_path: Path, sample_raw_df: pd.DataFrame) -> Path:
    csv_path = tmp_path / "corazon_sample.csv"
    sample_raw_df.to_csv(csv_path, index=False)
    return csv_path


def _build_valid_rows(n: int) -> pd.DataFrame:
    """Build n rows that are individually valid against RAW_SCHEMA and the
    heart-rate integrity rule, so tests can control exactly how many rows
    (and what fraction) get corrupted afterwards.
    """
    sexes = ["Male", "Female"]
    chest_pains = ["typical", "asymptomatic", "nonanginal", "nontypical"]
    rest_ecgs = ["normal", "left ventricular hypertrophy", "ST-T wave abnormality"]
    thals = ["normal", "fixed", "reversable"]

    rows = []
    for i in range(n):
        age = 40 + (i % 30)
        rows.append(
            {
                "age": age,
                "sex": sexes[i % 2],
                "chest_pain": chest_pains[i % 4],
                "rest_bp": 110 + (i % 40),
                "chol": 180 + (i % 100),
                "fbs": str(i % 2),
                "rest_ecg": rest_ecgs[i % 3],
                "max_hr": min(180, 220 - age - 5),
                "exang": str(i % 2),
                "old_peak": round((i % 5) * 0.4, 1),
                "slope": str((i % 3) + 1),
                "ca": str(i % 4),
                "thal": thals[i % 3],
                "disease": i % 2,
            }
        )
    return pd.DataFrame(rows)


def _write_csv(df: pd.DataFrame, tmp_path: Path, name: str) -> Path:
    csv_path = tmp_path / name
    df.to_csv(csv_path, index=False)
    return csv_path


# --------------------------------------------------------------------------
# load_raw_data
# --------------------------------------------------------------------------


def test_load_raw_data_drops_duplicates_and_coerces_types(sample_raw_csv: Path) -> None:
    df = load_raw_data(sample_raw_csv)

    # the second and third rows in the fixture are exact duplicates
    assert len(df) == EXPECTED_ROWS_AFTER_DEDUP
    for col in NUM_COLS:
        assert pd.api.types.is_numeric_dtype(df[col])
    for col in CAT_COLS:
        # Don't assert on the exact pandas dtype backend (object vs the
        # newer StringDtype) - both are correct. What matters is that
        # normalization produced plain Python strings (or NaN).
        non_null = df[col].dropna()
        assert non_null.map(lambda v: isinstance(v, str)).all()


def test_load_raw_data_raises_on_missing_columns(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"age": [1, 2]}).to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="missing expected columns"):
        load_raw_data(bad_csv)


def test_load_raw_data_drops_rows_with_invalid_target(tmp_path: Path, sample_raw_csv: Path) -> None:
    # Rows 1 and 2 of the fixture (index 1, 2) are an exact-duplicate pair -
    # corrupt rows 0 and 3 instead, so the duplicate pair stays intact and
    # dedup still collapses it as expected. Corrupt the CSV as text so we
    # never hit pandas' in-memory dtype strictness.
    lines = sample_raw_csv.read_text().splitlines()
    header = lines[0].split(",")
    disease_idx = header.index("disease")
    rows = [line.split(",") for line in lines[1:]]
    rows[0][disease_idx] = ""  # missing target
    rows[3][disease_idx] = "not_a_number"  # garbage target

    corrupted = tmp_path / "bad_target.csv"
    corrupted.write_text(",".join(header) + "\n" + "\n".join(",".join(r) for r in rows) + "\n")

    df = load_raw_data(corrupted)

    assert df[TARGET].isna().sum() == 0
    # original 5 rows -> 4 after dedup -> 2 more dropped for invalid target
    assert len(df) == EXPECTED_ROWS_AFTER_TARGET_FILTER


# --------------------------------------------------------------------------
# build_preprocessor
# --------------------------------------------------------------------------


def test_build_preprocessor_fits_and_transforms_without_nans(
    sample_raw_df: pd.DataFrame,
) -> None:
    preprocessor = build_preprocessor()
    features = sample_raw_df.drop(columns=[TARGET])

    transformed = preprocessor.fit_transform(features)

    assert transformed.shape[0] == len(features)
    assert not np.isnan(transformed).any()


def test_build_preprocessor_handles_unseen_category_at_inference(
    sample_raw_df: pd.DataFrame,
) -> None:
    preprocessor = build_preprocessor()
    features = sample_raw_df.drop(columns=[TARGET])
    preprocessor.fit(features)

    new_row = features.iloc[[0]].copy()
    new_row["thal"] = "never_seen_category"

    # must not raise thanks to handle_unknown="ignore"
    transformed = preprocessor.transform(new_row)
    assert transformed.shape[0] == 1


# --------------------------------------------------------------------------
# validate_raw_data (strict gate, no recovery)
# --------------------------------------------------------------------------


def test_validate_raw_data_passes_clean_data(sample_raw_csv: Path) -> None:
    df = load_raw_data(sample_raw_csv)
    df = clean_invalid_rows(df)

    # must not raise
    validate_raw_data(df)


def test_validate_raw_data_raises_when_null_rate_too_high(tmp_path: Path) -> None:
    df = _build_valid_rows(20)
    n_null = int(len(df) * (MAX_NULL_PCT + 0.2))
    df.loc[: n_null - 1, "chol"] = np.nan  # well above MAX_NULL_PCT
    csv_path = _write_csv(df, tmp_path, "too_many_nulls.csv")

    loaded = load_raw_data(csv_path)
    with pytest.raises(DataValidationError, match="max null rate"):
        validate_raw_data(loaded)


# --------------------------------------------------------------------------
# clean_invalid_rows (recoverable cleaning step, includes heart-rate rule)
# --------------------------------------------------------------------------


def test_clean_invalid_rows_drops_rows_below_invalid_threshold(tmp_path: Path) -> None:
    df = _build_valid_rows(20)
    df.loc[0, "sex"] = "9999"  # 1/20 = 5% -> below MAX_INVALID_ROW_FRACTION
    csv_path = _write_csv(df, tmp_path, "mostly_valid.csv")

    loaded = load_raw_data(csv_path)
    cleaned = clean_invalid_rows(loaded)

    assert len(cleaned) == len(loaded) - 1
    # cleaned data must now pass the strict validation gate
    validate_raw_data(cleaned)


def test_clean_invalid_rows_raises_when_too_many_rows_invalid(tmp_path: Path) -> None:
    df = _build_valid_rows(20)
    n_corrupt = int(len(df) * (MAX_INVALID_ROW_FRACTION + 0.2))
    df.loc[: n_corrupt - 1, "sex"] = "9999"  # well above MAX_INVALID_ROW_FRACTION
    csv_path = _write_csv(df, tmp_path, "mostly_invalid.csv")

    loaded = load_raw_data(csv_path)
    with pytest.raises(DataValidationError, match="failed schema checks"):
        clean_invalid_rows(loaded)


def test_clean_invalid_rows_drops_heart_rate_integrity_violations() -> None:
    df = _build_valid_rows(5)
    df.loc[0, "age"] = 70
    df.loc[0, "max_hr"] = 200  # 220 - 70 + 20 = 170 -> 200 violates the rule

    result = clean_invalid_rows(df)

    assert len(result) == len(df) - 1


# --------------------------------------------------------------------------
# run_feature_pipeline (end-to-end)
# --------------------------------------------------------------------------


def test_run_feature_pipeline_persists_features_and_preprocessor(
    tmp_path: Path,
) -> None:
    df = _build_valid_rows(20)
    csv_path = _write_csv(df, tmp_path, "corazon_sample.csv")
    features_path = tmp_path / "out" / "features.parquet"
    pipeline_path = tmp_path / "out" / "feature_pipeline.pkl"

    result_df = run_feature_pipeline(
        raw_path=csv_path,
        features_path=features_path,
        pipeline_path=pipeline_path,
    )

    assert features_path.exists()
    assert pipeline_path.exists()
    assert SPLIT_COL in result_df.columns
    assert set(result_df[SPLIT_COL].unique()) == {"train", "test"}

    persisted_df = pd.read_parquet(features_path)
    pd.testing.assert_frame_equal(
        result_df.reset_index(drop=True),
        persisted_df.reset_index(drop=True),
        check_dtype=False,
    )

    fitted_preprocessor = joblib.load(pipeline_path)
    raw_sample = df.drop(columns=[TARGET]).iloc[[0]]
    transformed = fitted_preprocessor.transform(raw_sample)
    assert transformed.shape[0] == 1


def test_run_feature_pipeline_raises_and_does_not_persist_on_invalid_data(
    tmp_path: Path,
) -> None:
    df = _build_valid_rows(20)
    n_null = int(len(df) * (MAX_NULL_PCT + 0.2))
    df.loc[: n_null - 1, "chol"] = np.nan  # breach the null-rate gate
    csv_path = _write_csv(df, tmp_path, "invalid.csv")

    features_path = tmp_path / "out" / "features.parquet"
    pipeline_path = tmp_path / "out" / "feature_pipeline.pkl"

    with pytest.raises(DataValidationError):
        run_feature_pipeline(
            raw_path=csv_path,
            features_path=features_path,
            pipeline_path=pipeline_path,
        )

    assert not features_path.exists()
    assert not pipeline_path.exists()
