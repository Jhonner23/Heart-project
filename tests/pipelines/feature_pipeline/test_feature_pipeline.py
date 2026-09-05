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
    SPLIT_COL,
    TARGET,
    build_preprocessor,
    load_raw_data,
    run_feature_pipeline,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Synthetic dataset shaped like corazon.csv, large enough to stratify-split,
    including edge cases: a duplicated row, a missing numeric value and a
    missing categorical value.
    """
    n = 40
    rng = np.random.default_rng(0)
    data = {
        "age": rng.integers(30, 80, n).astype(float),
        "sex": rng.choice(["Male", "Female"], n),
        "chest_pain": rng.choice(["typical", "asymptomatic", "nontypical"], n),
        "rest_bp": rng.integers(100, 180, n).astype(float),
        "chol": rng.integers(150, 300, n).astype(float),
        "fbs": rng.choice([0, 1], n),
        "rest_ecg": rng.choice(["normal", "left ventricular hypertrophy"], n),
        "max_hr": rng.integers(90, 200, n).astype(float),
        "exang": rng.choice([0, 1], n),
        "old_peak": rng.uniform(0, 4, n).round(1),
        "slope": rng.choice([1, 2, 3], n),
        "ca": rng.choice([0.0, 1.0, 2.0, 3.0], n),
        "thal": rng.choice(["normal", "fixed", "reversable"], n),
        "disease": rng.choice([0, 1], n),
    }
    df = pd.DataFrame(data)
    df.loc[1] = df.loc[0]  # exact duplicate row
    df.loc[2, "age"] = np.nan  # missing numeric
    df.loc[3, "thal"] = None  # missing categorical
    return df


@pytest.fixture
def sample_raw_csv(tmp_path: Path, sample_raw_df: pd.DataFrame) -> Path:
    csv_path = tmp_path / "corazon_sample.csv"
    sample_raw_df.to_csv(csv_path, index=False)
    return csv_path


def test_load_raw_data_drops_duplicates_and_coerces_types(sample_raw_csv: Path) -> None:
    df = load_raw_data(sample_raw_csv)

    assert len(df) == len(pd.read_csv(sample_raw_csv).drop_duplicates())
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

    transformed = preprocessor.transform(new_row)
    assert transformed.shape[0] == 1


def test_run_feature_pipeline_persists_transformed_features(
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
    assert len(persisted_df) == len(result_df)
    assert TARGET in persisted_df.columns
    assert SPLIT_COL in persisted_df.columns
    assert set(persisted_df[SPLIT_COL].unique()) <= {"train", "test"}

    # the persisted matrix must actually be transformed: no raw category
    # strings should remain, every non-target/split column must be numeric.
    feature_cols = [c for c in persisted_df.columns if c not in (TARGET, SPLIT_COL)]
    assert len(feature_cols) > len(NUM_COLS) + len(CAT_COLS)  # one-hot expanded the cat columns
    for col in feature_cols:
        assert pd.api.types.is_numeric_dtype(persisted_df[col])
    assert not persisted_df[feature_cols].isna().any().any()


def test_run_feature_pipeline_avoids_leakage(tmp_path: Path, sample_raw_csv: Path) -> None:
    """The persisted preprocessor must be fit on train rows only: transforming
    the test rows with a preprocessor fit on train-only data must match what
    was actually persisted for those same rows."""
    features_path = tmp_path / "out" / "features.parquet"
    pipeline_path = tmp_path / "out" / "feature_pipeline.pkl"

    run_feature_pipeline(
        raw_path=sample_raw_csv,
        features_path=features_path,
        pipeline_path=pipeline_path,
    )

    persisted_df = pd.read_parquet(features_path)
    fitted_preprocessor = joblib.load(pipeline_path)

    raw_df = load_raw_data(sample_raw_csv)
    train_mask = (persisted_df[SPLIT_COL] == "train").to_numpy()
    reference_preprocessor = build_preprocessor()
    reference_preprocessor.fit(raw_df.drop(columns=[TARGET]).loc[train_mask])

    test_features = raw_df.drop(columns=[TARGET]).loc[~train_mask]
    expected = reference_preprocessor.transform(test_features)
    actual = fitted_preprocessor.transform(test_features)

    np.testing.assert_allclose(actual, expected)
