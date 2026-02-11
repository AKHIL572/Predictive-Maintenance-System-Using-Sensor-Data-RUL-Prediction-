import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "train_FD001.csv"
MODEL_PATH = BASE_DIR / "Models" / "rf_rul_model.pkl"
FEATURE_PATH = BASE_DIR / "Models" / "feature_columns.pkl"

# ---------------- LOAD ARTIFACTS ----------------


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, features


df = load_data()
model, feature_cols = load_model()

# ---------------- SENSOR GROUPING ----------------
trend_sensors = ["sensor_7", "sensor_9", "sensor_12"]
important_sensors = ["sensor_7", "sensor_9", "sensor_12", "sensor_4"]

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Predictive Maintenance - RUL",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔧 Predictive Maintenance (RUL)")
st.caption("Predict engine health using sensor data")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Engine Selection")
engine_ids = sorted(df["engine_id"].unique())
engine_id = st.sidebar.selectbox("Select Engine ID", engine_ids)

engine_df = df[df["engine_id"] == engine_id]
last_cycle = engine_df["cycle"].max()
latest_row = engine_df[engine_df["cycle"] == last_cycle]

# ---------------- PREDICTION ----------------
X_latest = latest_row[feature_cols]
predicted_rul = int(model.predict(X_latest)[0])

# ---------------- STATUS ----------------
if predicted_rul <= 10:
    status = "CRITICAL"
elif predicted_rul <= 40:
    status = "WARNING"
else:
    status = "HEALTHY"

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted RUL", f"{predicted_rul} cycles")

with col2:
    st.metric("Last Cycle", last_cycle)

with col3:
    st.metric("Engine Status", status)

st.divider()

# ---------------- SENSOR TREND ----------------
st.subheader("📉 Sensor Degradation Trend")
sensor_to_plot = st.selectbox("Select Sensor", important_sensors)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(engine_df["cycle"], engine_df[sensor_to_plot])
ax.set_xlabel("Cycle")
ax.set_ylabel(sensor_to_plot)
ax.set_title(f"{sensor_to_plot} Trend for Engine {engine_id}")
st.pyplot(fig)

# ---------------- SENSOR INSIGHT ----------------
st.markdown("### 🔍 Sensor Insight")

if sensor_to_plot in trend_sensors:
    st.info(
        f"""
        **Observation:**  
        `{sensor_to_plot}` shows a clear degradation pattern as the engine approaches failure.
        This sensor is strongly correlated with Remaining Useful Life (RUL), making it a
        reliable health indicator.
        """
    )
elif sensor_to_plot == "sensor_4":
    st.warning(
        """
        **Observation:**  
        `sensor_4` is highly important to the Random Forest model but does not show a smooth,
        monotonic degradation trend.
        
        The model uses this sensor through non-linear thresholds and interactions rather
        than direct time-based degradation.
        """
    )
else:
    st.warning(
        f"""
        `{sensor_to_plot}` provides contextual information but has weaker direct correlation
        with Remaining Useful Life.
        """
    )

# ---------------- LATEST SENSOR VALUES ----------------
st.subheader("📄 View Latest Sensor Values")

latest_sensors = latest_row[important_sensors].T
latest_sensors.columns = ["Value"]

st.dataframe(latest_sensors)

csv = latest_sensors.to_csv().encode("utf-8")
st.download_button(
    label="⬇️ Download Latest Sensor Values (CSV)",
    data=csv,
    file_name=f"engine_{engine_id}_latest_sensors.csv",
    mime="text/csv"
)
