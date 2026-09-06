"""Training pipeline: trains and evaluates the heart-disease classifier.

Reads the transformed feature table produced by ``feature_pipeline.py``
(``data/04_feature/corazon_features.parquet``), reuses the ``split`` column
already persisted there (train/test rows were fixed once, upstream, so this
script never re-splits or re-randomizes), verifies that the train/test
separation itself is sound, trains a classifier, evaluates it on the
held-out test rows, and persists the fitted model plus both sets of results.

Model choice: ``RandomForestClassifier`` with the hyperparameters selected
in Trabajo 1's model-selection notebook
(notebooks/6-interpretation/06.Seleccion_Modelo-*.ipynb), where it was
compared against Logistic Regression, Gradient Boosting and SVM via 5-fold
cross-validation plus a GridSearchCV hyperparameter search, and won on F1.
This script re-implements that decision as a standalone, reproducible
production step rather than re-running the model comparison.

Train/test split checks (``validate_train_test_split``): mirrors the checks
in a Deepchecks-style ``train_test_validation`` suite (see
https://joserzapata.github.io/courses/ciencia-datos-en-produccion/data-validation/train_test-checks/)
without adding that dependency, since the project already favors small,
explicit pandas/scipy checks (see ``feature_pipeline.py``'s validation) over
heavier frameworks:
- Index leakage (fatal): no row index may appear in both ``X_train`` and
  ``X_test``. This is the one unambiguous definition of leakage - it means
  the same source record was literally used for both training and
  evaluation - so it raises ``TrainTestSplitError`` and blocks training.
  ``split_train_test`` can't actually produce this today (each row has a
  single ``split`` value), so this check exists to catch a *future*
  regression, not today's data.
- Sample mixing (warning): the fraction of test rows that are exact
  duplicates, feature-for-feature, of a train row. On real clinical data
  with missing values, this is expected, not necessarily a bug: several
  columns here have a meaningful null rate (see ``feature_pipeline.py``'s
  ``SimpleImputer`` step), and mean/median imputation deterministically
  fills every missing cell in a column with the same value - so two
  genuinely different patients who already matched on their non-missing
  fields can become identical feature vectors purely from imputation, with
  no partitioning bug involved. Verified against this project's own data:
  0 duplicate rows exist before imputation/encoding, but ~12% of test rows
  duplicate a train row afterwards. Treating that as fatal would block
  training on legitimate data, so it's reported as a warning instead.
- Feature drift: a two-sample Kolmogorov-Smirnov test per feature flags
  columns whose train/test distributions differ significantly. Some drift
  is expected from sampling variance in a ~460-row dataset, so this is a
  warning, not a fatal error.
- Label drift: the positive-class rate should be similar in train and
  test (this dataset's split is already stratified, so this mostly acts as
  a regression check that stratification didn't silently break). Also a
  warning, not fatal.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from scipy import stats
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

# Train/test split-check thresholds (see module docstring).
MAX_TRAIN_TEST_OVERLAP: float = 0.05
DRIFT_KS_PVALUE_THRESHOLD: float = 0.01
MAX_DRIFTED_FEATURE_FRACTION: float = 0.3
MAX_LABEL_DRIFT: float = 0.15

# How many leaked indices to list in the error message before truncating.
MAX_LEAKED_INDICES_SHOWN: int = 10

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


class TrainTestSplitError(ValueError):
    """Raised when the train/test split itself is unsound (e.g. data leakage)."""


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


def validate_train_test_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Check that the train/test split is sound: no leakage, similar distributions.

    Independent and testable on its own (doesn't need a fitted model): pass
    it any X_train/X_test/y_train/y_test and it returns a results dict.

    Raises ``TrainTestSplitError`` - blocking training entirely - only if the
    same row index appears in both splits (unambiguous leakage: the same
    source record was used for both training and evaluation). Sample mixing
    (exact-duplicate feature vectors across splits), feature drift and label
    drift are reported as warnings in the returned dict (and logged), not
    fatal - see the module docstring for why exact-duplicate feature vectors
    are expected on this dataset and aren't proof of a partitioning bug.
    """
    results: dict[str, Any] = {"warnings": []}

    # --- Index leakage (fatal): the same row must never be in both splits ---
    shared_index = X_train.index.intersection(X_test.index)
    if len(shared_index) > 0:
        shown = list(shared_index)[:MAX_LEAKED_INDICES_SHOWN]
        suffix = "..." if len(shared_index) > MAX_LEAKED_INDICES_SHOWN else ""
        raise TrainTestSplitError(
            f"{len(shared_index)} row index/indices appear in both train and "
            f"test: {shown}{suffix}. This means the same source record was "
            "used for both training and evaluation."
        )

    # --- Sample mixing: exact-duplicate feature vectors across the split ---
    combined = pd.concat([X_train, X_test])
    is_duplicate = combined.duplicated(keep=False)
    n_leaking_test_rows = int(is_duplicate.loc[X_test.index].sum())
    overlap_fraction = n_leaking_test_rows / len(X_test) if len(X_test) else 0.0
    results["overlap_fraction"] = overlap_fraction

    if overlap_fraction > MAX_TRAIN_TEST_OVERLAP:
        warning = (
            f"{overlap_fraction:.1%} of test rows are exact duplicates, "
            f"feature-for-feature, of a train row (expected up to "
            f"{MAX_TRAIN_TEST_OVERLAP:.0%}). Verify this comes from "
            "imputation collapsing distinct records, not a partitioning bug."
        )
        logger.warning(warning)
        results["warnings"].append(warning)

    # --- Feature drift: per-column two-sample Kolmogorov-Smirnov test ---
    drifted_features = []
    for col in X_train.columns:
        p_value = stats.ks_2samp(X_train[col], X_test[col]).pvalue
        if p_value < DRIFT_KS_PVALUE_THRESHOLD:
            drifted_features.append(col)
    drifted_fraction = len(drifted_features) / len(X_train.columns) if len(X_train.columns) else 0.0
    results["drifted_features"] = drifted_features
    results["drifted_feature_fraction"] = drifted_fraction

    if drifted_fraction > MAX_DRIFTED_FEATURE_FRACTION:
        warning = (
            f"{drifted_fraction:.1%} of features show a significant train/test "
            f"distribution shift (KS test, p<{DRIFT_KS_PVALUE_THRESHOLD}): "
            f"{drifted_features}"
        )
        logger.warning(warning)
        results["warnings"].append(warning)

    # --- Label drift: positive-class rate should be similar in both splits ---
    train_positive_rate = float(y_train.mean())
    test_positive_rate = float(y_test.mean())
    label_drift = abs(train_positive_rate - test_positive_rate)
    results["train_positive_rate"] = train_positive_rate
    results["test_positive_rate"] = test_positive_rate
    results["label_drift"] = label_drift

    if label_drift > MAX_LABEL_DRIFT:
        warning = (
            f"Label drift between train ({train_positive_rate:.1%} positive) and "
            f"test ({test_positive_rate:.1%} positive) is {label_drift:.1%} "
            f"(max expected: {MAX_LABEL_DRIFT:.0%})"
        )
        logger.warning(warning)
        results["warnings"].append(warning)

    if not results["warnings"]:
        logger.info(
            "Train/test split checks passed: overlap=%.1f%%, drifted_features=%d/%d, "
            "label_drift=%.1f%%",
            overlap_fraction * 100,
            len(drifted_features),
            len(X_train.columns),
            label_drift * 100,
        )
    return results


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
) -> dict[str, Any]:
    """Evaluate the fitted classifier on the held-out test split.

    Includes accuracy, precision, recall, F1 and AUC-ROC - appropriate for
    a binary classification problem where, in this clinical context, missed
    positive cases (false negatives) and false alarms both carry real cost,
    so no single metric alone is enough.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics: dict[str, Any] = {
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
) -> dict[str, Any]:
    """Run the full training pipeline: load -> split -> validate split ->
    train -> evaluate -> persist.

    Nothing is trained or persisted if ``validate_train_test_split`` raises
    ``TrainTestSplitError`` (the same row index found in both train and
    test).
    """
    logger.info("Loading features from %s", features_path)
    df = load_features(features_path)

    X_train, X_test, y_train, y_test = split_train_test(df)
    split_validation = validate_train_test_split(X_train, X_test, y_train, y_test)

    model = build_model()
    model = train_model(model, X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    metrics["train_test_validation"] = split_validation

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
