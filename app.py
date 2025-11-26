import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================
# CONFIGURACIÓN INICIAL
# ==========================
st.set_page_config(
    page_title="Predicción de precios Airbnb - Buenos Aires",
    page_icon="🏙️",
    layout="centered"
)

st.title("🏙️ Predicción de precios de alojamientos Airbnb (Buenos Aires)")
st.write(
    "Ingresá las características del alojamiento y obtené el **precio estimado por noche** "
    "según un modelo entrenado sobre datos reales de Airbnb."
)

MODEL_PATH = Path("modelo_lightgbm_mejor.joblib")

# ==========================
# BARRIOS CODIFICADOS (TARGET ENCODING)
# ==========================
BARRIO_TO_CODE = {
    "Almagro": 374606.52875882946,
    "Balvanera": 363425.516888434,
    "Barracas": 436464.41441441444,
    "Belgrano": 479570.16628873773,
    "Boca": 395666.875,
    "Boedo": 327596.96629213484,
    "Caballito": 383833.6653386454,
    "Chacarita": 405955.7403189066,
    "Coghlan": 356747.2619047619,
    "Colegiales": 515163.4923339012,
    "Constitucion": 365756.775147929,
    "Flores": 360640.0,
    "Monserrat": 392797.72402854875,
    "Núñez": 497182.1082089552,
    "Otros": 358104.89285714284,
    "Palermo": 526028.5508110687,
    "Parque Chacabuco": 377125.9016393443,
    "Parque Chas": 370419.01960784313,
    "Parque Patricios": 365979.3548387097,
    "Puerto Madero": 978665.1079136691,
    "Recoleta": 494501.60500260553,
    "Retiro": 480087.18600953894,
    "Saavedra": 396603.77659574465,
    "San Cristóbal": 343419.74358974356,
    "San Nicolás": 418487.3333333333,
    "San Telmo": 410064.1815856777,
    "Villa Crespo": 409395.1871657754,
    "Villa del Parque": 383260.0,
    "Villa Devoto": 452844.1176470588,
    "Villa Ortúzar": 355644.1304347826,
    "Villa Pueyrredón": 376421.4,
    "Villa Urquiza": 408502.8213166144,
}

# ==========================
# CARGA DEL MODELO
# ==========================
@st.cache_resource
def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path.resolve()}")
    return joblib.load(model_path)

try:
    model = load_model(MODEL_PATH)
    model_loaded = True
except Exception as e:
    st.error("❌ No se pudo cargar el modelo.")
    st.code(str(e))
    model_loaded = False

# ==========================
# INPUTS DEL USUARIO
# ==========================
st.sidebar.header("✏️ Parámetros del alojamiento")

accommodates = st.sidebar.number_input("Capacidad (personas)", min_value=1, value=2, step=1)

bedrooms = st.sidebar.number_input("Dormitorios", min_value=0, value=1, step=1)

beds = st.sidebar.number_input("Camas", min_value=0, value=1, step=1)

bathrooms = st.sidebar.number_input("Baños", min_value=0.0, value=1.0, step=0.5)

# Estas dos sí tienen máximo
availability_365 = st.sidebar.number_input(
    "Disponibilidad anual (1–365 días)", 
    min_value=1, 
    max_value=365, 
    value=180
)

review_scores_rating = st.sidebar.number_input(
    "Puntaje reviews (1–5)", 
    min_value=1.0, 
    max_value=5.0, 
    value=4.5, 
    step=0.1
)

# Estas NO tienen límite máximo
reviews_per_month = st.sidebar.number_input("Reviews por mes", min_value=0.0, value=1.0, step=0.1)

number_of_reviews = st.sidebar.number_input("Número total de reviews", min_value=0, value=10, step=1)

dist_obelisco_km = st.sidebar.number_input("Distancia al Obelisco (km)", min_value=0.0, value=5.0, step=0.1)


# Dropdown con los barrios
barrio = st.sidebar.selectbox(
    "Barrio",
    options=sorted(BARRIO_TO_CODE.keys())
)

barrio_encoded = float(BARRIO_TO_CODE[barrio])

# ==========================
# ARMAR DATAFRAME PARA EL MODELO
# ==========================
FEATURES = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "availability_365",
    "review_scores_rating",
    "reviews_per_month",
    "dist_obelisco_km",
    "number_of_reviews",
    "barrio_encoded",
]

input_data = pd.DataFrame([{
    "accommodates": accommodates,
    "bedrooms": bedrooms,
    "beds": beds,
    "bathrooms": bathrooms,
    "availability_365": availability_365,
    "review_scores_rating": review_scores_rating,
    "reviews_per_month": reviews_per_month,
    "dist_obelisco_km": dist_obelisco_km,
    "number_of_reviews": number_of_reviews,
    "barrio_encoded": barrio_encoded,
}])[FEATURES]

with st.expander("🔍 Ver datos enviados al modelo"):
    st.dataframe(input_data)

# ==========================
# PREDICCIÓN
# ==========================
st.markdown("---")
st.subheader("📈 Predicción")

if st.button("Predecir precio por noche"):
    if not model_loaded:
        st.error("No se pudo cargar el modelo.")
    else:
        try:
            pred = model.predict(input_data)[0]
            st.success(f"💵 Precio estimado por noche: **ARS {pred:,.2f}**")
        except Exception as e:
            st.error("❌ Error al hacer la predicción.")
            st.code(str(e))
