"""Training pipeline: trains and evaluates the heart-disease classifier.

Reads the transformed feature table produced by ``feature_pipeline.py``
(``data/04_feature/corazon_features.parquet``), reuses the ``split`` column
already persisted there (train/test rows were fixed once, upstream, so this
script never re-splits or re-randomizes), trains a classifier, evaluates it
on the held-out test rows, and persists both the fitted model and the
evaluation metrics.

Model choice: ``RandomForestClassifier`` with the hyperparameters selected
in Trabajo 1's model-selection notebook
(notebooks/6-interpretation/06.Seleccion_Modelo-*.ipynb), where it was
compared against Logistic Regression, Gradient Boosting and SVM via 5-fold
cross-validation plus a GridSearchCV hyperparameter search, and won on F1.
This script re-implements that decision as a standalone, reproducible
production step rather than re-running the model comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

TARGET: str = "disease"
SPLIT_COL: str = "split"

RANDOM_STATE: int = 42

# Hyperparameters selected via GridSearchCV in Trabajo 1's model-selection
# notebook (best F1 among Logistic Regression / Random Forest / Gradient
# Boosting / SVM).
MODEL_PARAMS: dict[str, int | None] = {
    "n_estimators": 200,
    "max_depth": 5,
    "min_samples_split": 2,
    "random_state": RANDOM_STATE,
}

ROOT_DIR: Path = Path(__file__).resolve().parents[3]
DEFAULT_FEATURES_PATH: Path = ROOT_DIR / "data" / "04_feature" / "corazon_features.parquet"
DEFAULT_MODEL_PATH: Path = ROOT_DIR / "models" / "heart_disease_classifier.pkl"
DEFAULT_METRICS_PATH: Path = ROOT_DIR / "models" / "train_metrics.json"


class TrainingDataError(ValueError):
    """Raised when the feature table is missing what the training step needs."""


def load_features(features_path: Path) -> pd.DataFrame:
    """Load the transformed feature table and check it has what training needs."""
    df = pd.read_parquet(features_path)

    missing_cols = {TARGET, SPLIT_COL} - set(df.columns)
    if missing_cols:
        raise TrainingDataError(
            f"Feature table is missing expected columns: {sorted(missing_cols)}. "
            "Did it come from feature_pipeline.py?"
        )
    return df


def split_train_test(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train/test using the ``split`` column persisted upstream.

    The split itself is NOT redone here: feature_pipeline.py already fixed
    it once (with a fixed random_state) when it fit the preprocessor, so
    re-splitting here would risk a different partition than the one the
    preprocessor's train-only statistics were actually fit on.
    """
    valid_splits = {"train", "test"}
    seen_splits = set(df[SPLIT_COL].unique())
    unexpected = seen_splits - valid_splits
    if unexpected:
        raise TrainingDataError(
            f"Unexpected values in '{SPLIT_COL}' column: {sorted(unexpected)} "
            f"(expected only {sorted(valid_splits)})"
        )
    missing_splits = valid_splits - seen_splits
    if missing_splits:
        raise TrainingDataError(
            f"Feature table is missing rows for split(s): {sorted(missing_splits)}"
        )

    feature_cols = [c for c in df.columns if c not in {TARGET, SPLIT_COL}]
    train_df = df.loc[df[SPLIT_COL] == "train"]
    test_df = df.loc[df[SPLIT_COL] == "test"]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[TARGET]
    y_test = test_df[TARGET]
    return X_train, X_test, y_train, y_test


def build_model() -> RandomForestClassifier:
    """Build the (unfitted) classifier with the selected hyperparameters."""
    return RandomForestClassifier(**MODEL_PARAMS)


def train_model(
    model: RandomForestClassifier, X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    """Fit the classifier on the training split only."""
    logger.info("Training %s on %d rows", type(model).__name__, len(X_train))
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float | list[list[int]]]:
    """Evaluate the fitted classifier on the held-out test split.

    Includes accuracy, precision, recall, F1 and AUC-ROC - appropriate for
    a binary classification problem where, in this clinical context, missed
    positive cases (false negatives) and false alarms both carry real cost,
    so no single metric alone is enough.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics: dict[str, float | list[list[int]]] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_test_samples": len(y_test),
    }
    logger.info(
        "Test metrics: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
    )
    return metrics


def run_train_pipeline(
    features_path: Path = DEFAULT_FEATURES_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> dict[str, float | list[list[int]]]:
    """Run the full training pipeline: load -> split -> train -> evaluate -> persist."""
    logger.info("Loading features from %s", features_path)
    df = load_features(features_path)

    X_train, X_test, y_train, y_test = split_train_test(df)

    model = build_model()
    model = train_model(model, X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    logger.info("Saved trained model to %s", model_path)
    logger.info("Saved evaluation metrics to %s", metrics_path)
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the training pipeline.")
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    run_train_pipeline(
        features_path=args.features_path,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
    )


if __name__ == "__main__":
    main()
