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
    MODEL_PARAMS,
    SPLIT_COL,
    TARGET,
    TrainingDataError,
    build_model,
    evaluate_model,
    load_features,
    run_train_pipeline,
    split_train_test,
    train_model,
)

EXPECTED_METRIC_KEYS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "confusion_matrix",
    "n_test_samples",
}
N_TRAIN_ROWS = 60
N_TEST_ROWS = 20
CONFUSION_MATRIX_SIZE = 2


def _build_features_df(n_train: int = N_TRAIN_ROWS, n_test: int = N_TEST_ROWS) -> pd.DataFrame:
    """Build a synthetic, already-transformed feature table shaped like the
    real output of feature_pipeline.py: numeric feature columns, a binary
    target that's actually learnable from the features, and a split column.
    """
    rng = np.random.default_rng(42)
    n = n_train + n_test

    feature_1 = rng.normal(size=n)
    feature_2 = rng.normal(size=n)
    # target correlated with feature_1 so the model has something real to learn
    target = (feature_1 + rng.normal(scale=0.5, size=n) > 0).astype(int)

    split = np.array(["train"] * n_train + ["test"] * n_test)
    rng.shuffle(split)

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

    assert set(metrics.keys()) == EXPECTED_METRIC_KEYS
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
