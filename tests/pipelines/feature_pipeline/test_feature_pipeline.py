"""Unit tests for the feature pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.pipelines.feature_pipeline.feature_pipeline import (
    CAT_COLS,
    NUM_COLS,
    TARGET,
    build_preprocessor,
    load_raw_data,
    run_feature_pipeline,
)


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


def test_load_raw_data_drops_duplicates_and_coerces_types(
    sample_raw_csv: Path, sample_raw_df: pd.DataFrame
) -> None:
    df = load_raw_data(sample_raw_csv)

    # the fixture has exactly one exact-duplicate row (rows 1 and 2)
    expected_unique_rows = sample_raw_df.drop_duplicates().shape[0]
    assert len(df) == expected_unique_rows
    for col in NUM_COLS:
        assert pd.api.types.is_numeric_dtype(df[col])
    for col in CAT_COLS:
        assert df[col].dtype == object


def test_load_raw_data_raises_on_missing_columns(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"age": [1, 2]}).to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="missing expected columns"):
        load_raw_data(bad_csv)


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


def test_run_feature_pipeline_persists_features_and_preprocessor(
    tmp_path: Path, sample_raw_csv: Path
) -> None:
    features_path = tmp_path / "out" / "features.parquet"
    pipeline_path = tmp_path / "out" / "feature_pipeline.pkl"

    result_df = run_feature_pipeline(
        raw_path=sample_raw_csv,
        features_path=features_path,
        pipeline_path=pipeline_path,
    )

    assert features_path.exists()
    assert pipeline_path.exists()

    persisted_df = pd.read_parquet(features_path)
    # check_dtype=False: parquet round-trips object columns as pandas'
    # arrow-backed StringDtype, which is a storage detail, not a pipeline bug.
    pd.testing.assert_frame_equal(
        result_df.reset_index(drop=True),
        persisted_df.reset_index(drop=True),
        check_dtype=False,
    )

    fitted_preprocessor = joblib.load(pipeline_path)
    sample_features = persisted_df.drop(columns=[TARGET]).iloc[[0]]
    transformed = fitted_preprocessor.transform(sample_features)
    assert transformed.shape[0] == 1
