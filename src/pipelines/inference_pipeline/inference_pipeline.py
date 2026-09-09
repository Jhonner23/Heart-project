"""Inference pipeline: applies the trained model to new, unlabeled data.

Loads the already-fitted preprocessor and the already-trained classifier
from project storage (both persisted by the feature and training
pipelines), reads new raw data from a file, applies the SAME fitted
transformations used at training time (``.transform`` only - never
``.fit``/``.fit_transform``, so no statistics are relearned from the new
data), generates predictions, and persists them alongside the original
rows.

Unlike the feature pipeline's ``load_raw_data``, the loader here does not
require (and never touches) the ``disease`` target column or the
``split`` column: at inference time the target is exactly what is being
predicted, and there is no train/test split to record.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.pipelines.feature_pipeline.feature_pipeline import (
    CAT_COLS,
    NUM_COLS,
    VALID_CATEGORIES,
    normalize_categorical,
)

logger = logging.getLogger(__name__)

PREDICTION_COL: str = "predicted_disease"
PREDICTION_PROBA_COL: str = "predicted_disease_probability"

ROOT_DIR: Path = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH: Path = ROOT_DIR / "models" / "heart_disease_classifier.pkl"
DEFAULT_PREPROCESSOR_PATH: Path = ROOT_DIR / "models" / "feature_pipeline.pkl"
DEFAULT_PREDICTIONS_PATH: Path = ROOT_DIR / "data" / "07_model_output" / "predictions.csv"


class InferenceDataError(ValueError):
    """Raised when new data is missing feature columns the model needs."""


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Any:
    """Load the trained classifier persisted by the training pipeline."""
    logger.info("Loading trained model from %s", model_path)
    return joblib.load(model_path)


def load_preprocessor(
    preprocessor_path: Path = DEFAULT_PREPROCESSOR_PATH,
) -> ColumnTransformer:
    """Load the fitted preprocessor persisted by the feature pipeline."""
    logger.info("Loading fitted preprocessor from %s", preprocessor_path)
    return joblib.load(preprocessor_path)


def _validate_categories(df: pd.DataFrame) -> None:
    """Reject categorical values the fitted preprocessor doesn't recognize.

    ``OneHotEncoder(handle_unknown="ignore")`` silently encodes an unknown
    category as all-zeros instead of raising - e.g. ``sex="male"`` (lowercase)
    would be encoded as neither Male nor Female, and the model would still
    happily return a prediction for that nonsense row with no warning at
    all. This checks every categorical column against the same
    ``VALID_CATEGORIES`` the training data was validated against, and fails
    loudly (blocking ALL predictions for the file) instead of letting a
    typo silently degrade a prediction. Missing values (NaN) are allowed
    through, matching the nullable categorical columns in
    ``feature_pipeline.RAW_SCHEMA``.
    """
    errors: list[str] = []
    for col in CAT_COLS:
        valid = set(VALID_CATEGORIES[col])
        observed = set(df[col].dropna().unique())
        invalid = observed - valid
        if invalid:
            errors.append(f"{col}: {sorted(invalid)} (valid: {sorted(valid)})")

    if errors:
        raise InferenceDataError(
            "New data has unrecognized categorical value(s):\n" + "\n".join(errors)
        )


def load_new_data(data_path: Path) -> pd.DataFrame:
    """Load new, unlabeled raw data for inference.

    Applies the same type coercion as ``feature_pipeline.load_raw_data``
    (numeric casting, categorical normalization) so the fitted preprocessor
    sees inputs shaped exactly like the training data, but requires only
    the feature columns - not the (unknown, to-be-predicted) target.

    Also validates every categorical value against ``VALID_CATEGORIES``
    (see ``_validate_categories``): a value the preprocessor's
    ``OneHotEncoder`` would otherwise ignore silently is instead a fatal,
    fully-described error here - no partial/degraded predictions.

    Rejects a file that already contains ``PREDICTION_COL``/
    ``PREDICTION_PROBA_COL`` (e.g. someone re-uploading a previous
    predictions output as if it were new input data): silently letting it
    through would concatenate a duplicate-named column onto the new
    predictions in ``run_inference_pipeline``, breaking every downstream
    ``result[PREDICTION_COL]`` lookup in a confusing way.
    """
    logger.info("Loading new data from %s", data_path)
    df = pd.read_csv(data_path)

    missing_cols = set(NUM_COLS + CAT_COLS) - set(df.columns)
    if missing_cols:
        raise InferenceDataError(
            f"New data is missing expected feature columns: {sorted(missing_cols)}"
        )

    reserved_cols = {PREDICTION_COL, PREDICTION_PROBA_COL} & set(df.columns)
    if reserved_cols:
        raise InferenceDataError(
            f"New data already contains reserved output column(s) {sorted(reserved_cols)} "
            "- make sure you're uploading the INPUT file (patient data), not a "
            "previous predictions output."
        )

    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in CAT_COLS:
        df[col] = df[col].map(normalize_categorical).astype(object)

    _validate_categories(df)

    return df


def transform_new_data(df: pd.DataFrame, preprocessor: ColumnTransformer) -> pd.DataFrame:
    """Apply the SAME already-fitted preprocessor used during training.

    Calls ``.transform`` only - the preprocessor was already fit on the
    training split by the feature pipeline, so this reuses its learned
    imputation/scaling/encoding statistics rather than relearning them
    from the new data (which would silently break train/inference parity).
    """
    transformed = preprocessor.transform(df[NUM_COLS + CAT_COLS])
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=feature_names, index=df.index)


def predict(model: Any, features: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions and, when the model supports it, probabilities."""
    predictions = model.predict(features)
    result = pd.DataFrame({PREDICTION_COL: predictions}, index=features.index)

    if hasattr(model, "predict_proba"):
        # last column: probability of the positive ("disease") class.
        result[PREDICTION_PROBA_COL] = model.predict_proba(features)[:, -1]

    return result


def run_inference_pipeline(
    data_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    preprocessor_path: Path = DEFAULT_PREPROCESSOR_PATH,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
) -> pd.DataFrame:
    """Run the full inference pipeline: load -> transform -> predict -> persist.

    Returns the original new-data rows with the prediction column(s)
    appended; the same table is written to ``predictions_path``.
    """
    model = load_model(model_path)
    preprocessor = load_preprocessor(preprocessor_path)
    df = load_new_data(data_path)

    features = transform_new_data(df, preprocessor)
    logger.info("Applied fitted preprocessor to %d rows (%d features)", *features.shape)

    predictions = predict(model, features)
    logger.info(
        "Generated %d predictions (%d flagged positive)",
        len(predictions),
        int(predictions[PREDICTION_COL].sum()),
    )

    result = pd.concat([df.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(predictions_path, index=False)
    logger.info("Saved predictions to %s", predictions_path)

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the inference pipeline on new data.")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="CSV file with new, unlabeled data (must contain the model's feature columns).",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--preprocessor-path", type=Path, default=DEFAULT_PREPROCESSOR_PATH)
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    run_inference_pipeline(
        data_path=args.data_path,
        model_path=args.model_path,
        preprocessor_path=args.preprocessor_path,
        predictions_path=args.predictions_path,
    )


if __name__ == "__main__":
    main()
