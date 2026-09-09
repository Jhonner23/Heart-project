"""Streamlit app: predicción de enfermedad cardíaca, online y por lotes.

Reutiliza tal cual la arquitectura FTI del proyecto: carga el modelo
entrenado y el preprocesador ya ajustado (persistidos por
``train_pipeline.py`` y ``feature_pipeline.py`` respectivamente) y aplica
las mismas funciones de ``inference_pipeline.py`` (``transform_new_data``,
``predict``) tanto para una predicción individual como para un archivo
completo, sin reimplementar esa lógica aquí.

Dos pestañas:
- "Predicción individual": formulario con los datos de un paciente.
- "Procesamiento batch": sube un CSV con varios pacientes y descarga las
  predicciones.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.pipelines.feature_pipeline.feature_pipeline import (
    CAT_COLS,
    NUM_COLS,
    NUMERIC_RANGES,
    VALID_CATEGORIES,
)
from src.pipelines.inference_pipeline.inference_pipeline import (
    PREDICTION_COL,
    PREDICTION_PROBA_COL,
    InferenceDataError,
    load_model,
    load_new_data,
    load_preprocessor,
    predict,
    transform_new_data,
)

ROOT_DIR = Path(__file__).parent
EXAMPLE_INPUT_PATH = ROOT_DIR / "examples" / "pacientes_nuevos_ejemplo.csv"
EXAMPLE_OUTPUT_PATH = ROOT_DIR / "examples" / "predicciones_ejemplo.csv"

# Etiquetas legibles para las categorías cuyo código no es autoexplicativo.
CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "fbs": {"0": "No", "1": "Sí"},
    "slope": {"1": "Ascendente", "2": "Plana", "3": "Descendente"},
    "exang": {"0": "No", "1": "Sí"},
}

FIELD_LABELS: dict[str, str] = {
    "age": "Edad (años)",
    "rest_bp": "Presión arterial en reposo (mmHg)",
    "chol": "Colesterol sérico (mg/dl)",
    "max_hr": "Frecuencia cardíaca máxima",
    "old_peak": "Depresión ST inducida (old_peak)",
    "sex": "Sexo",
    "chest_pain": "Tipo de dolor de pecho",
    "fbs": "Glucosa en ayunas > 120 mg/dl",
    "rest_ecg": "Resultado ECG en reposo",
    "slope": "Pendiente del segmento ST",
    "ca": "Vasos principales coloreados (fluoroscopía)",
    "thal": "Talasemia",
    "exang": "Angina inducida por ejercicio",
}

st.set_page_config(
    page_title="Predicción Enfermedad Cardíaca",
    page_icon="❤️",
    layout="centered",
)


@st.cache_resource
def get_model() -> object:
    return load_model()


@st.cache_resource
def get_preprocessor() -> object:
    return load_preprocessor()


model = get_model()
preprocessor = get_preprocessor()

st.title("❤️ Predicción de Enfermedad Cardíaca")
st.caption(
    "Usa el mismo modelo Random Forest y el mismo preprocesador ajustados por "
    "el pipeline de entrenamiento del proyecto (arquitectura FTI: Feature -> "
    "Training -> Inference). Herramienta de apoyo; no reemplaza el "
    "diagnóstico médico profesional."
)

tab_online, tab_batch = st.tabs(["🔍 Predicción individual", "📁 Procesamiento batch"])

# ── Pestaña 1: predicción individual (online) ────────────────────────────────
with tab_online:
    st.markdown(
        "**Instrucciones:** completa los datos del paciente y presiona "
        "**Predecir** para obtener el resultado del modelo."
    )

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            FIELD_LABELS["age"],
            min_value=int(NUMERIC_RANGES["age"][0]),
            max_value=int(NUMERIC_RANGES["age"][1]),
            value=50,
        )
        rest_bp = st.number_input(
            FIELD_LABELS["rest_bp"],
            min_value=int(NUMERIC_RANGES["rest_bp"][0]),
            max_value=int(NUMERIC_RANGES["rest_bp"][1]),
            value=120,
        )
        chol = st.number_input(
            FIELD_LABELS["chol"],
            min_value=int(NUMERIC_RANGES["chol"][0]),
            max_value=int(NUMERIC_RANGES["chol"][1]),
            value=240,
        )
        max_hr = st.number_input(
            FIELD_LABELS["max_hr"],
            min_value=int(NUMERIC_RANGES["max_hr"][0]),
            max_value=int(NUMERIC_RANGES["max_hr"][1]),
            value=150,
        )
        old_peak = st.number_input(
            FIELD_LABELS["old_peak"],
            min_value=float(NUMERIC_RANGES["old_peak"][0]),
            max_value=float(NUMERIC_RANGES["old_peak"][1]),
            value=1.0,
            step=0.1,
        )
        exang = st.selectbox(
            FIELD_LABELS["exang"],
            options=VALID_CATEGORIES["exang"],
            format_func=lambda x: CATEGORY_LABELS["exang"][x],
        )

    with col2:
        sex = st.selectbox(FIELD_LABELS["sex"], options=VALID_CATEGORIES["sex"])
        chest_pain = st.selectbox(
            FIELD_LABELS["chest_pain"], options=VALID_CATEGORIES["chest_pain"]
        )
        fbs = st.selectbox(
            FIELD_LABELS["fbs"],
            options=VALID_CATEGORIES["fbs"],
            format_func=lambda x: CATEGORY_LABELS["fbs"][x],
        )
        rest_ecg = st.selectbox(FIELD_LABELS["rest_ecg"], options=VALID_CATEGORIES["rest_ecg"])
        slope = st.selectbox(
            FIELD_LABELS["slope"],
            options=VALID_CATEGORIES["slope"],
            format_func=lambda x: CATEGORY_LABELS["slope"][x],
        )
        ca = st.selectbox(FIELD_LABELS["ca"], options=VALID_CATEGORIES["ca"])
        thal = st.selectbox(FIELD_LABELS["thal"], options=VALID_CATEGORIES["thal"])

    if st.button("🔍 Predecir", type="primary"):
        input_data = pd.DataFrame(
            [
                {
                    "age": float(age),
                    "rest_bp": float(rest_bp),
                    "chol": float(chol),
                    "max_hr": float(max_hr),
                    "old_peak": float(old_peak),
                    "sex": sex,
                    "chest_pain": chest_pain,
                    "fbs": fbs,
                    "rest_ecg": rest_ecg,
                    "slope": slope,
                    "ca": ca,
                    "thal": thal,
                    "exang": exang,
                }
            ]
        )
        for col in CAT_COLS:
            input_data[col] = input_data[col].astype(object)

        features = transform_new_data(input_data, preprocessor)
        result = predict(model, features)
        pred = result.loc[0, PREDICTION_COL]
        prob = result.loc[0, PREDICTION_PROBA_COL]

        st.divider()
        if pred == 1:
            st.error(f"⚠️ **Resultado: Enfermedad cardíaca detectada** (probabilidad: {prob:.1%})")
        else:
            st.success(
                f"✅ **Resultado: Sin enfermedad cardíaca** (probabilidad de enfermedad: {prob:.1%})"
            )

# ── Pestaña 2: procesamiento batch ───────────────────────────────────────────
with tab_batch:
    st.markdown(
        "**Instrucciones:** sube un archivo CSV con una fila por paciente y "
        f"las columnas {', '.join(NUM_COLS + CAT_COLS)} (sin la columna "
        "`disease`, que es justamente lo que se va a predecir). Obtendrás una "
        "tabla con dos columnas nuevas: `predicted_disease` (0/1) y "
        "`predicted_disease_probability`."
    )

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        if EXAMPLE_INPUT_PATH.exists():
            st.download_button(
                "⬇️ Descargar CSV de ejemplo (entrada)",
                data=EXAMPLE_INPUT_PATH.read_bytes(),
                file_name=EXAMPLE_INPUT_PATH.name,
                mime="text/csv",
            )
    with dl_col2:
        if EXAMPLE_OUTPUT_PATH.exists():
            st.download_button(
                "⬇️ Descargar CSV de ejemplo (salida esperada)",
                data=EXAMPLE_OUTPUT_PATH.read_bytes(),
                file_name=EXAMPLE_OUTPUT_PATH.name,
                mime="text/csv",
            )

    st.divider()

    uploaded_file = st.file_uploader("Sube tu archivo CSV de pacientes nuevos", type="csv")

    if uploaded_file is not None:
        # load_new_data espera una ruta de archivo (misma función que usa el
        # CLI de inferencia): se guarda el archivo subido en un temporal para
        # reutilizarla sin reimplementar su validación/coerción de tipos.
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)

        try:
            new_data = load_new_data(tmp_path)
            features = transform_new_data(new_data, preprocessor)
            predictions = predict(model, features)
            result = pd.concat(
                [new_data.reset_index(drop=True), predictions.reset_index(drop=True)],
                axis=1,
            )
        except InferenceDataError as exc:
            st.error(f"⚠️ El archivo no tiene el formato esperado: {exc}")
        else:
            n_positive = int(result[PREDICTION_COL].sum())
            st.success(f"Se generaron {len(result)} predicciones ({n_positive} positivas).")
            st.dataframe(result, use_container_width=True)
            st.download_button(
                "⬇️ Descargar predicciones",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="predicciones.csv",
                mime="text/csv",
            )
        finally:
            tmp_path.unlink(missing_ok=True)
