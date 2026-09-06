"""Unit tests for the training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.pipelines.training_pipeline.train_pipeline import (
    MAX_LABEL_DRIFT,
    MAX_TRAIN_TEST_OVERLAP,
    MODEL_PARAMS,
    SPLIT_COL,
    TARGET,
    TrainingDataError,
    TrainTestSplitError,
    build_model,
    evaluate_model,
    load_features,
    run_train_pipeline,
    split_train_test,
    train_model,
    validate_train_test_split,
)

EXPECTED_METRIC_KEYS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "confusion_matrix",
    "n_test_samples",
    "train_test_validation",
}
EXPECTED_SPLIT_VALIDATION_KEYS = {
    "warnings",
    "overlap_fraction",
    "drifted_features",
    "drifted_feature_fraction",
    "train_positive_rate",
    "test_positive_rate",
    "label_drift",
}
CONFUSION_MATRIX_SIZE = 2
N_TRAIN_ROWS = 60
N_TEST_ROWS = 20


def _build_features_df(n_train: int = N_TRAIN_ROWS, n_test: int = N_TEST_ROWS) -> pd.DataFrame:
    """Build a synthetic, already-transformed feature table shaped like the
    real output of feature_pipeline.py: numeric feature columns, a binary
    target that's actually learnable from the features, and a split column.

    The split is assigned per target class (stratified), mirroring the real
    upstream split in feature_pipeline.py. This keeps the positive-class rate
    close between train and test by construction, instead of leaving it to
    chance the way an unstratified shuffle would - which would make
    label-drift assertions flaky.
    """
    rng = np.random.default_rng(42)
    n = n_train + n_test

    feature_1 = rng.normal(size=n)
    feature_2 = rng.normal(size=n)
    # target correlated with feature_1 so the model has something real to learn
    target = (feature_1 + rng.normal(scale=0.5, size=n) > 0).astype(int)

    test_fraction = n_test / n
    split = np.array(["train"] * n)
    for label in (0, 1):
        class_idx = np.flatnonzero(target == label)
        n_test_for_class = round(len(class_idx) * test_fraction)
        test_idx = rng.choice(class_idx, size=n_test_for_class, replace=False)
        split[test_idx] = "test"

    return pd.DataFrame(
        {
            "num__feature_1": feature_1,
            "num__feature_2": feature_2,
            TARGET: target,
            SPLIT_COL: split,
        }
    )


def _write_parquet(df: pd.DataFrame, tmp_path: Path, name: str = "features.parquet") -> Path:
    path = tmp_path / name
    df.to_parquet(path, index=False)
    return path


# --------------------------------------------------------------------------
# load_features
# --------------------------------------------------------------------------


def test_load_features_reads_valid_table(tmp_path: Path) -> None:
    df = _build_features_df()
    path = _write_parquet(df, tmp_path)

    loaded = load_features(path)

    assert len(loaded) == len(df)
    assert TARGET in loaded.columns
    assert SPLIT_COL in loaded.columns


def test_load_features_raises_when_target_missing(tmp_path: Path) -> None:
    df = _build_features_df().drop(columns=[TARGET])
    path = _write_parquet(df, tmp_path)

    with pytest.raises(TrainingDataError, match="missing expected columns"):
        load_features(path)


def test_load_features_raises_when_split_column_missing(tmp_path: Path) -> None:
    df = _build_features_df().drop(columns=[SPLIT_COL])
    path = _write_parquet(df, tmp_path)

    with pytest.raises(TrainingDataError, match="missing expected columns"):
        load_features(path)


# --------------------------------------------------------------------------
# split_train_test
# --------------------------------------------------------------------------


def test_split_train_test_uses_persisted_split_column() -> None:
    df = _build_features_df()

    X_train, X_test, y_train, y_test = split_train_test(df)

    assert len(X_train) == N_TRAIN_ROWS
    assert len(X_test) == N_TEST_ROWS
    assert len(y_train) == N_TRAIN_ROWS
    assert len(y_test) == N_TEST_ROWS
    # feature matrices must not leak the target or the split column
    assert TARGET not in X_train.columns
    assert SPLIT_COL not in X_train.columns


def test_split_train_test_raises_on_unexpected_split_value() -> None:
    df = _build_features_df()
    df.loc[df.index[0], SPLIT_COL] = "validation"

    with pytest.raises(TrainingDataError, match="Unexpected values"):
        split_train_test(df)


def test_split_train_test_raises_when_a_split_is_entirely_missing() -> None:
    df = _build_features_df()
    df[SPLIT_COL] = "train"  # no test rows at all

    with pytest.raises(TrainingDataError, match="missing rows for split"):
        split_train_test(df)


# --------------------------------------------------------------------------
# validate_train_test_split
# --------------------------------------------------------------------------


def test_validate_train_test_split_passes_on_a_clean_iid_split() -> None:
    df = _build_features_df(n_train=200, n_test=100)
    X_train, X_test, y_train, y_test = split_train_test(df)

    results = validate_train_test_split(X_train, X_test, y_train, y_test)

    assert set(results.keys()) == EXPECTED_SPLIT_VALIDATION_KEYS
    assert results["warnings"] == []
    assert results["overlap_fraction"] == 0.0


def test_validate_train_test_split_raises_on_shared_index() -> None:
    df = _build_features_df(n_train=200, n_test=100)
    X_train, X_test, y_train, y_test = split_train_test(df)

    # Force real leakage: make a test row share its index with a train row.
    X_test = X_test.copy()
    shared_index = X_train.index[0]
    X_test = X_test.rename(index={X_test.index[0]: shared_index})
    y_test = y_test.rename(index={y_test.index[0]: shared_index})

    with pytest.raises(TrainTestSplitError, match="both train and test"):
        validate_train_test_split(X_train, X_test, y_train, y_test)


def test_validate_train_test_split_warns_on_duplicated_feature_rows() -> None:
    df = _build_features_df(n_train=200, n_test=100)
    X_train, X_test, y_train, y_test = split_train_test(df)

    # Duplicate feature vectors across the split (e.g. what imputation can
    # cause on real data), without touching the index - not leakage on its
    # own, just worth a warning.
    n_duplicated = int(len(X_test) * (MAX_TRAIN_TEST_OVERLAP + 0.2))
    X_test = X_test.copy()
    X_test.iloc[:n_duplicated] = X_train.iloc[:n_duplicated].to_numpy()

    results = validate_train_test_split(X_train, X_test, y_train, y_test)

    assert results["overlap_fraction"] > MAX_TRAIN_TEST_OVERLAP
    assert any("exact duplicates" in w for w in results["warnings"])


def test_validate_train_test_split_warns_on_feature_drift() -> None:
    df = _build_features_df(n_train=200, n_test=100)
    X_train, X_test, y_train, y_test = split_train_test(df)

    # Shift every test feature far away from the train distribution.
    X_test = X_test + 10.0

    results = validate_train_test_split(X_train, X_test, y_train, y_test)

    assert results["drifted_feature_fraction"] > 0
    assert any("distribution shift" in w for w in results["warnings"])


def test_validate_train_test_split_warns_on_label_drift() -> None:
    df = _build_features_df(n_train=200, n_test=100)
    X_train, X_test, y_train, y_test = split_train_test(df)

    # Force an extreme class-balance mismatch between train and test.
    y_train = pd.Series([0] * len(y_train), index=y_train.index)
    y_test = pd.Series([1] * len(y_test), index=y_test.index)

    results = validate_train_test_split(X_train, X_test, y_train, y_test)

    assert results["label_drift"] > MAX_LABEL_DRIFT
    assert any("Label drift" in w for w in results["warnings"])


# --------------------------------------------------------------------------
# build_model / train_model
# --------------------------------------------------------------------------


def test_build_model_uses_selected_hyperparameters() -> None:
    model = build_model()

    assert isinstance(model, RandomForestClassifier)
    for param, value in MODEL_PARAMS.items():
        assert getattr(model, param) == value


def test_train_model_returns_a_fitted_estimator() -> None:
    df = _build_features_df()
    X_train, _, y_train, _ = split_train_test(df)
    model = build_model()

    fitted = train_model(model, X_train, y_train)

    # a fitted RandomForestClassifier exposes trained estimators
    assert hasattr(fitted, "estimators_")
    assert len(fitted.estimators_) == MODEL_PARAMS["n_estimators"]


# --------------------------------------------------------------------------
# evaluate_model
# --------------------------------------------------------------------------


def test_evaluate_model_returns_expected_keys_and_valid_ranges() -> None:
    df = _build_features_df()
    X_train, X_test, y_train, y_test = split_train_test(df)
    model = train_model(build_model(), X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    assert set(metrics.keys()) == EXPECTED_METRIC_KEYS - {"train_test_validation"}
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        value = metrics[key]
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0
    assert metrics["n_test_samples"] == N_TEST_ROWS

    confusion = metrics["confusion_matrix"]
    assert isinstance(confusion, list)
    assert len(confusion) == CONFUSION_MATRIX_SIZE
    assert len(confusion[0]) == CONFUSION_MATRIX_SIZE


# --------------------------------------------------------------------------
# run_train_pipeline (end-to-end)
# --------------------------------------------------------------------------


def test_run_train_pipeline_persists_model_and_metrics(tmp_path: Path) -> None:
    df = _build_features_df()
    features_path = _write_parquet(df, tmp_path)
    model_path = tmp_path / "out" / "model.pkl"
    metrics_path = tmp_path / "out" / "metrics.json"

    metrics = run_train_pipeline(
        features_path=features_path,
        model_path=model_path,
        metrics_path=metrics_path,
    )

    assert model_path.exists()
    assert metrics_path.exists()

    persisted_model = joblib.load(model_path)
    assert isinstance(persisted_model, RandomForestClassifier)

    persisted_metrics = json.loads(metrics_path.read_text())
    assert persisted_metrics == metrics
    assert set(persisted_metrics.keys()) == EXPECTED_METRIC_KEYS
    assert set(persisted_metrics["train_test_validation"].keys()) == (
        EXPECTED_SPLIT_VALIDATION_KEYS
    )


def test_run_train_pipeline_raises_and_does_not_persist_on_bad_input(
    tmp_path: Path,
) -> None:
    df = _build_features_df().drop(columns=[SPLIT_COL])
    features_path = _write_parquet(df, tmp_path)
    model_path = tmp_path / "out" / "model.pkl"
    metrics_path = tmp_path / "out" / "metrics.json"

    with pytest.raises(TrainingDataError):
        run_train_pipeline(
            features_path=features_path,
            model_path=model_path,
            metrics_path=metrics_path,
        )

    assert not model_path.exists()
    assert not metrics_path.exists()


def test_run_train_pipeline_persists_and_warns_on_duplicated_feature_rows(
    tmp_path: Path,
) -> None:
    df = _build_features_df(n_train=200, n_test=100)
    # Duplicate a chunk of train rows' features into test (matching target
    # values), as distinct rows with their own index - like imputation
    # collapsing distinct patients to the same feature vector on real data.
    # This is NOT index leakage, so the pipeline should still complete and
    # just surface a warning, not raise.
    train_rows = df[df[SPLIT_COL] == "train"].iloc[:50].copy()
    train_rows[SPLIT_COL] = "test"
    duplicated_df = pd.concat([df, train_rows], ignore_index=True)

    features_path = _write_parquet(duplicated_df, tmp_path)
    model_path = tmp_path / "out" / "model.pkl"
    metrics_path = tmp_path / "out" / "metrics.json"

    metrics = run_train_pipeline(
        features_path=features_path,
        model_path=model_path,
        metrics_path=metrics_path,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    split_validation = metrics["train_test_validation"]
    assert split_validation["overlap_fraction"] > MAX_TRAIN_TEST_OVERLAP
    assert any("exact duplicates" in w for w in split_validation["warnings"])
