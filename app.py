from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ── Configuración ────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "models" / "06_svm_model.joblib"

st.set_page_config(
    page_title="Predicción Enfermedad Cardíaca",
    page_icon="❤️",
    layout="centered",
)


@st.cache_resource
def load_model() -> object:
    return joblib.load(MODEL_PATH)


pipeline = load_model()

# ── Interfaz ─────────────────────────────────────────────────────────────────
st.title("❤️ Predicción de Enfermedad Cardíaca")
st.markdown("Ingresa los datos del paciente para obtener la predicción del modelo SVM entrenado.")

st.header("Datos del paciente")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Edad (años)", min_value=20, max_value=100, value=50)
    rest_bp = st.number_input(
        "Presión arterial en reposo (mmHg)", min_value=80, max_value=220, value=120
    )
    chol = st.number_input("Colesterol sérico (mg/dl)", min_value=100, max_value=600, value=240)
    max_hr = st.number_input("Frecuencia cardíaca máxima", min_value=60, max_value=220, value=150)
    old_peak = st.number_input(
        "Depresión ST inducida (old_peak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )

with col2:
    sex = st.selectbox(
        "Sexo",
        options=[0, 1],
        format_func=lambda x: "Femenino" if x == 0 else "Masculino",
    )
    chest_pain = st.selectbox(
        "Tipo de dolor de pecho",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Angina típica",
            2: "Angina atípica",
            3: "No anginoso",
            4: "Asintomático",
        }[x],
    )
    fbs = st.selectbox(
        "Glucosa en ayunas > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Sí",
    )
    rest_ecg = st.selectbox(
        "Resultado ECG en reposo",
        options=[0, 1, 2],
        format_func=lambda x: {0: "Normal", 1: "Anormalidad ST-T", 2: "Hipertrofia VI"}[x],
    )
    slope = st.selectbox(
        "Pendiente del segmento ST",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Ascendente", 2: "Plana", 3: "Descendente"}[x],
    )
    ca = st.selectbox("Vasos principales coloreados (fluoroscopía)", options=[0, 1, 2, 3])
    thal = st.selectbox(
        "Talasemia",
        options=[3, 6, 7],
        format_func=lambda x: {3: "Normal", 6: "Defecto fijo", 7: "Defecto reversible"}[x],
    )

# ── Predicción ────────────────────────────────────────────────────────────────
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
            }
        ]
    )

    # Convertir tipos para compatibilidad con el pipeline
    num_cols = ["age", "rest_bp", "chol", "max_hr", "old_peak"]
    cat_cols = ["sex", "chest_pain", "fbs", "rest_ecg", "slope", "ca", "thal"]
    for col in num_cols:
        input_data[col] = input_data[col].astype(float)
    for col in cat_cols:
        input_data[col] = input_data[col].astype(object)

    pred = pipeline.predict(input_data)[0]
    prob = pipeline.predict_proba(input_data)[0][1]

    st.divider()
    if pred == 1:
        st.error(f"⚠️ **Resultado: Enfermedad cardíaca detectada** (probabilidad: {prob:.1%})")
    else:
        st.success(
            f"✅ **Resultado: Sin enfermedad cardíaca** (probabilidad de enfermedad: {prob:.1%})"
        )

    st.caption(
        "Este modelo es una herramienta de apoyo y no reemplaza el diagnóstico médico profesional."
    )
