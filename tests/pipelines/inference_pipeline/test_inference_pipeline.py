"""Unit tests for the inference pipeline.

These tests never load the real, trained RandomForest or the real fitted
preprocessor. Instead they use a dummy classifier (``sklearn.dummy.
DummyClassifier``) and a preprocessor fitted on small synthetic data that
mirrors the project's real feature schema (``NUM_COLS``/``CAT_COLS``), so
the tests stay fast and fully isolated from the actual model/data
pipelines while still exercising the real loading, transformation and
prediction code paths.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier

from src.pipelines.feature_pipeline.feature_pipeline import (
    CAT_COLS,
    NUM_COLS,
    build_preprocessor,
)
from src.pipelines.inference_pipeline.inference_pipeline import (
    PREDICTION_COL,
    PREDICTION_PROBA_COL,
    InferenceDataError,
    load_model,
    load_new_data,
    load_preprocessor,
    predict,
    run_inference_pipeline,
    transform_new_data,
)

VALID_ROWS: dict[str, list[object]] = {
    "age": [63, 45, 58, 50, 67, 41],
    "rest_bp": [145, 130, 120, 140, 160, 110],
    "chol": [233, 210, 240, 200, 286, 190],
    "max_hr": [150, 165, 140, 155, 108, 170],
    "old_peak": [2.3, 0.5, 1.0, 0.0, 1.5, 0.2],
    "sex": ["Male", "Female", "Male", "Female", "Male", "Female"],
    "chest_pain": [
        "typical",
        "asymptomatic",
        "nonanginal",
        "nontypical",
        "asymptomatic",
        "typical",
    ],
    "fbs": ["1", "0", "0", "1", "0", "0"],
    "rest_ecg": [
        "left ventricular hypertrophy",
        "normal",
        "normal",
        "ST-T wave abnormality",
        "left ventricular hypertrophy",
        "normal",
    ],
    "slope": ["3", "1", "2", "1", "2", "1"],
    "ca": ["0", "1", "0", "2", "3", "0"],
    "thal": ["fixed", "normal", "normal", "reversable", "normal", "fixed"],
    "exang": ["0", "1", "0", "0", "1", "0"],
}


@pytest.fixture
def synthetic_new_data() -> pd.DataFrame:
    return pd.DataFrame(VALID_ROWS)


@pytest.fixture
def fitted_preprocessor(synthetic_new_data: pd.DataFrame) -> ColumnTransformer:
    preprocessor = build_preprocessor()
    preprocessor.fit(synthetic_new_data[NUM_COLS + CAT_COLS])
    return preprocessor


@pytest.fixture
def dummy_model(
    synthetic_new_data: pd.DataFrame, fitted_preprocessor: ColumnTransformer
) -> DummyClassifier:
    features = transform_new_data(synthetic_new_data, fitted_preprocessor)
    labels = [0, 1, 0, 1, 1, 0]
    model = DummyClassifier(strategy="stratified", random_state=42)
    model.fit(features, labels)
    return model


class TestLoadNewData:
    def test_loads_and_coerces_types(
        self, tmp_path: Path, synthetic_new_data: pd.DataFrame
    ) -> None:
        data_path = tmp_path / "new_data.csv"
        synthetic_new_data.to_csv(data_path, index=False)

        df = load_new_data(data_path)

        assert len(df) == len(synthetic_new_data)
        for col in NUM_COLS:
            assert df[col].dtype == float
        # normalize_categorical: an integer-valued float string like "1.0"
        # collapses to "1", matching what training-time data looked like.
        assert df.loc[0, "fbs"] == "1"

    def test_missing_feature_columns_raise(
        self, tmp_path: Path, synthetic_new_data: pd.DataFrame
    ) -> None:
        incomplete = synthetic_new_data.drop(columns=["chol", "thal"])
        data_path = tmp_path / "incomplete.csv"
        incomplete.to_csv(data_path, index=False)

        with pytest.raises(InferenceDataError, match="chol"):
            load_new_data(data_path)

    def test_does_not_require_target_or_split_columns(
        self, tmp_path: Path, synthetic_new_data: pd.DataFrame
    ) -> None:
        # No 'disease' (target) and no 'split' column: both are legitimately
        # absent from new, unlabeled inference data.
        assert "disease" not in synthetic_new_data.columns
        assert "split" not in synthetic_new_data.columns
        data_path = tmp_path / "new_data.csv"
        synthetic_new_data.to_csv(data_path, index=False)

        df = load_new_data(data_path)

        assert "disease" not in df.columns
        assert "split" not in df.columns

    def test_rejects_unrecognized_category_value(
        self, tmp_path: Path, synthetic_new_data: pd.DataFrame
    ) -> None:
        # "male" (lowercase) is not in VALID_CATEGORIES["sex"] ("Male"/
        # "Female"): OneHotEncoder(handle_unknown="ignore") would otherwise
        # silently zero out both sex columns instead of raising, so this
        # must be caught here, before the data ever reaches the model.
        bad = synthetic_new_data.copy()
        bad.loc[0, "sex"] = "male"
        data_path = tmp_path / "bad_category.csv"
        bad.to_csv(data_path, index=False)

        with pytest.raises(InferenceDataError, match="sex"):
            load_new_data(data_path)

    def test_allows_missing_category_values(
        self, tmp_path: Path, synthetic_new_data: pd.DataFrame
    ) -> None:
        # A missing (NaN) categorical value is not an "unrecognized
        # category" - it mirrors the nullable categorical columns in
        # feature_pipeline.RAW_SCHEMA and must pass through untouched.
        missing = synthetic_new_data.copy()
        missing.loc[0, "thal"] = None
        data_path = tmp_path / "missing_category.csv"
        missing.to_csv(data_path, index=False)

        df = load_new_data(data_path)

        assert pd.isna(df.loc[0, "thal"])

    def test_rejects_file_that_already_has_prediction_columns(
        self, tmp_path: Path, synthetic_new_data: pd.DataFrame
    ) -> None:
        # Someone re-uploading a previous predictions output as if it were
        # new input data: silently accepting it would later concat a
        # duplicate-named column onto the new predictions and break every
        # result[PREDICTION_COL] lookup downstream in a confusing way.
        looks_like_output = synthetic_new_data.copy()
        looks_like_output[PREDICTION_COL] = 0
        looks_like_output[PREDICTION_PROBA_COL] = 0.1
        data_path = tmp_path / "looks_like_output.csv"
        looks_like_output.to_csv(data_path, index=False)

        with pytest.raises(InferenceDataError, match=PREDICTION_COL):
            load_new_data(data_path)


class TestTransformNewData:
    def test_output_shape_matches_preprocessor(
        self, synthetic_new_data: pd.DataFrame, fitted_preprocessor: ColumnTransformer
    ) -> None:
        transformed = transform_new_data(synthetic_new_data, fitted_preprocessor)

        assert len(transformed) == len(synthetic_new_data)
        assert list(transformed.columns) == list(fitted_preprocessor.get_feature_names_out())

    def test_reuses_fitted_statistics_without_refitting(
        self, synthetic_new_data: pd.DataFrame, fitted_preprocessor: ColumnTransformer
    ) -> None:
        # A single-row batch, transformed independently, must use the SAME
        # (already-fitted) scaling/imputation statistics as the full batch -
        # proof that transform_new_data never calls .fit/.fit_transform.
        single_row = synthetic_new_data.iloc[[0]]
        transformed_full = transform_new_data(synthetic_new_data, fitted_preprocessor)
        transformed_single = transform_new_data(single_row, fitted_preprocessor)

        pd.testing.assert_frame_equal(
            transformed_single.reset_index(drop=True),
            transformed_full.iloc[[0]].reset_index(drop=True),
        )


class TestPredict:
    def test_predict_includes_probability_when_supported(
        self,
        synthetic_new_data: pd.DataFrame,
        fitted_preprocessor: ColumnTransformer,
        dummy_model: DummyClassifier,
    ) -> None:
        features = transform_new_data(synthetic_new_data, fitted_preprocessor)

        result = predict(dummy_model, features)

        assert PREDICTION_COL in result.columns
        assert PREDICTION_PROBA_COL in result.columns
        assert len(result) == len(features)
        assert result[PREDICTION_COL].isin([0, 1]).all()
        assert result[PREDICTION_PROBA_COL].between(0, 1).all()

    def test_predict_without_predict_proba(
        self, synthetic_new_data: pd.DataFrame, fitted_preprocessor: ColumnTransformer
    ) -> None:
        class PredictOnlyModel:
            def predict(self, X: pd.DataFrame) -> list[int]:
                return [0] * len(X)

        features = transform_new_data(synthetic_new_data, fitted_preprocessor)

        result = predict(PredictOnlyModel(), features)

        assert PREDICTION_COL in result.columns
        assert PREDICTION_PROBA_COL not in result.columns


class TestLoadModelAndPreprocessor:
    def test_load_model_round_trip(self, tmp_path: Path, dummy_model: DummyClassifier) -> None:
        model_path = tmp_path / "dummy_model.pkl"
        joblib.dump(dummy_model, model_path)

        loaded = load_model(model_path)

        assert hasattr(loaded, "predict")

    def test_load_preprocessor_round_trip(
        self, tmp_path: Path, fitted_preprocessor: ColumnTransformer
    ) -> None:
        preprocessor_path = tmp_path / "dummy_preprocessor.pkl"
        joblib.dump(fitted_preprocessor, preprocessor_path)

        loaded = load_preprocessor(preprocessor_path)

        assert list(loaded.get_feature_names_out()) == list(
            fitted_preprocessor.get_feature_names_out()
        )


class TestRunInferencePipeline:
    def test_end_to_end_with_dummy_model(
        self,
        tmp_path: Path,
        synthetic_new_data: pd.DataFrame,
        fitted_preprocessor: ColumnTransformer,
        dummy_model: DummyClassifier,
    ) -> None:
        data_path = tmp_path / "new_data.csv"
        synthetic_new_data.to_csv(data_path, index=False)

        model_path = tmp_path / "model.pkl"
        preprocessor_path = tmp_path / "preprocessor.pkl"
        joblib.dump(dummy_model, model_path)
        joblib.dump(fitted_preprocessor, preprocessor_path)

        predictions_path = tmp_path / "output" / "predictions.csv"

        result = run_inference_pipeline(
            data_path=data_path,
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            predictions_path=predictions_path,
        )

        assert predictions_path.exists()
        assert len(result) == len(synthetic_new_data)
        assert PREDICTION_COL in result.columns
        assert PREDICTION_PROBA_COL in result.columns
        # original feature columns are preserved alongside the predictions.
        for col in NUM_COLS + CAT_COLS:
            assert col in result.columns

        saved = pd.read_csv(predictions_path)
        assert len(saved) == len(synthetic_new_data)
        assert PREDICTION_COL in saved.columns

    def test_raises_on_missing_feature_columns(
        self,
        tmp_path: Path,
        synthetic_new_data: pd.DataFrame,
        fitted_preprocessor: ColumnTransformer,
        dummy_model: DummyClassifier,
    ) -> None:
        incomplete = synthetic_new_data.drop(columns=["age"])
        data_path = tmp_path / "incomplete.csv"
        incomplete.to_csv(data_path, index=False)

        model_path = tmp_path / "model.pkl"
        preprocessor_path = tmp_path / "preprocessor.pkl"
        joblib.dump(dummy_model, model_path)
        joblib.dump(fitted_preprocessor, preprocessor_path)

        with pytest.raises(InferenceDataError):
            run_inference_pipeline(
                data_path=data_path,
                model_path=model_path,
                preprocessor_path=preprocessor_path,
                predictions_path=tmp_path / "predictions.csv",
            )
