import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "train_FD001.csv"
MODEL_PATH = BASE_DIR / "models" / "rf_rul_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_columns.pkl"

# ---------------- SENSOR CONFIG ----------------
trend_sensors = ["sensor_7", "sensor_9", "sensor_12"]
important_sensors = ["sensor_7", "sensor_9", "sensor_12", "sensor_4",
                     "sensor_11", "sensor_14", "sensor_15"]

# ---------------- PAGE CONFIG (must be first st call) ----------------
st.set_page_config(
    page_title="Predictive Maintenance - RUL",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD ARTIFACTS ----------------


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not FEATURE_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_PATH)
    return model, feature_cols


# ---------------- LOAD & VALIDATE ----------------
df = load_data()
model, feature_cols = load_model()

if df is None:
    st.error("Dataset not found. Make sure `data/train_FD001.csv` exists.")
    st.stop()

if model is None:
    st.error("Model not found. Run `train.py` first to generate the model files.")
    st.stop()

missing_features = [c for c in feature_cols if c not in df.columns]
if missing_features:
    st.error(
        f"The following feature columns are missing from the dataset: {missing_features}")
    st.stop()

# Only keep sensors that actually exist in the dataset
important_sensors = [s for s in important_sensors if s in df.columns]

# ---------------- TITLE ----------------
st.title("🔧 Predictive Maintenance — RUL Estimator")
st.caption(
    "Monitor engine health and predict remaining useful life from sensor data.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Engine Selection")

engine_ids = sorted(df["engine_id"].unique())
engine_id = st.sidebar.selectbox("Select Engine ID", engine_ids)

engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")
min_cycle = int(engine_df["cycle"].min())
max_cycle = int(engine_df["cycle"].max())

# Cycle slider — lets user inspect any point in the engine's life
# Default to 75% through the engine lifetime so the first view is not always CRITICAL
default_cycle = min_cycle + int((max_cycle - min_cycle) * 0.75)
selected_cycle = st.sidebar.slider(
    "Inspect at Cycle",
    min_value=min_cycle,
    max_value=max_cycle,
    value=default_cycle,
    help="Move the slider to see predicted RUL and health status at any point "
         "in the engine's operational history."
)

selected_row = engine_df[engine_df["cycle"] == selected_cycle]

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Engine {engine_id} ran for **{max_cycle} cycles** total.  \n"
    f"Inspecting cycle **{selected_cycle}** of {max_cycle}."
)

# ---------------- PREDICTION ----------------
X_selected = selected_row[feature_cols]
predicted_rul = int(model.predict(X_selected)[0])

# Actual RUL from data (ground truth for comparison)
actual_rul = int(selected_row["RUL"].values[0]) if "RUL" in selected_row.columns else None

# ---------------- STATUS ----------------
if predicted_rul <= 10:
    status = "CRITICAL"
    status_color = "error"
elif predicted_rul <= 40:
    status = "WARNING"
    status_color = "warning"
else:
    status = "HEALTHY"
    status_color = "success"

# ---------------- METRICS ----------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Predicted RUL", f"{predicted_rul} cycles")

with col2:
    if actual_rul is not None:
        delta = predicted_rul - actual_rul
        st.metric("Actual RUL", f"{actual_rul} cycles",
                  delta=f"{delta:+d} cycles", delta_color="inverse")
    else:
        st.metric("Actual RUL", "N/A")

with col3:
    st.metric("Selected Cycle", f"{selected_cycle} / {max_cycle}")

with col4:
    st.metric("Engine Status", status)

if status_color == "error":
    st.error(
        f"🚨 Engine {engine_id} is in **CRITICAL** condition at cycle {selected_cycle} — "
        f"maintenance required immediately (predicted RUL: {predicted_rul} cycles).")
elif status_color == "warning":
    st.warning(
        f"⚠️ Engine {engine_id} is showing **WARNING** signs at cycle {selected_cycle} — "
        f"schedule maintenance soon (predicted RUL: {predicted_rul} cycles).")
else:
    st.success(
        f"✅ Engine {engine_id} is **HEALTHY** at cycle {selected_cycle} — "
        f"no immediate action required (predicted RUL: {predicted_rul} cycles).")

st.divider()

# ---------------- RUL TRAJECTORY ----------------
st.subheader("📈 RUL Prediction Trajectory")

# Predict RUL for every cycle of this engine to show full trajectory
all_X = engine_df[feature_cols]
engine_df = engine_df.copy()
engine_df["predicted_rul"] = model.predict(all_X)

fig_traj, ax_traj = plt.subplots(figsize=(10, 4))
ax_traj.plot(engine_df["cycle"], engine_df["predicted_rul"],
             label="Predicted RUL", linewidth=2, color="#1f77b4")
if "RUL" in engine_df.columns:
    ax_traj.plot(engine_df["cycle"], engine_df["RUL"],
                 label="Actual RUL", linewidth=1.5,
                 linestyle="--", color="#ff7f0e", alpha=0.8)

# Mark selected cycle
ax_traj.axvline(x=selected_cycle, color="red", linestyle=":",
                linewidth=1.5, label=f"Selected cycle ({selected_cycle})")

# Status threshold lines
ax_traj.axhline(y=40, color="orange", linestyle="--", linewidth=1, alpha=0.6, label="Warning threshold (40)")
ax_traj.axhline(y=10, color="red",    linestyle="--", linewidth=1, alpha=0.6, label="Critical threshold (10)")

ax_traj.set_xlabel("Cycle")
ax_traj.set_ylabel("RUL (cycles)")
ax_traj.set_title(f"RUL Trajectory — Engine {engine_id}")
ax_traj.legend(fontsize=8)
ax_traj.grid(True, alpha=0.3)
plt.tight_layout()

st.pyplot(fig_traj)
plt.close(fig_traj)

st.divider()

# ---------------- SENSOR TREND ----------------
st.subheader("📉 Sensor Degradation Trend")

sensor_to_plot = st.selectbox("Select Sensor", important_sensors)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(engine_df["cycle"], engine_df[sensor_to_plot], linewidth=1.5)
ax.axvline(x=selected_cycle, color="red", linestyle=":",
           linewidth=1.5, label=f"Selected cycle ({selected_cycle})")
ax.set_xlabel("Cycle")
ax.set_ylabel(sensor_to_plot)
ax.set_title(f"{sensor_to_plot} Trend — Engine {engine_id}")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# ---------------- SENSOR INSIGHT ----------------
st.markdown("### 🔍 Sensor Insight")

if sensor_to_plot in trend_sensors:
    st.info(
        f"**{sensor_to_plot}** shows a clear degradation pattern as the engine approaches "
        f"failure. It is strongly correlated with RUL and is a reliable health indicator."
    )
elif sensor_to_plot == "sensor_4":
    st.warning(
        "**sensor_4** is highly important to the Random Forest model but does not show a "
        "smooth monotonic trend. The model uses it through non-linear thresholds and feature "
        "interactions rather than direct time-based degradation."
    )
else:
    st.info(
        f"**{sensor_to_plot}** provides contextual information used by the model. "
        f"Its relationship with RUL may be non-linear or conditional on other features."
    )

# ---------------- LATEST SENSOR VALUES ----------------
st.subheader("📄 Sensor Readings at Selected Cycle")

display_sensors = [s for s in important_sensors if s in selected_row.columns]
latest_sensors = selected_row[display_sensors].T.reset_index()
latest_sensors.columns = ["Sensor", "Value"]
latest_sensors["Value"] = latest_sensors["Value"].round(4)

st.dataframe(latest_sensors, use_container_width=True, hide_index=True)

# ---------------- DOWNLOAD ----------------
csv = latest_sensors.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Sensor Values at Selected Cycle (CSV)",
    data=csv,
    file_name=f"engine_{engine_id}_cycle_{selected_cycle}_sensors.csv",
    mime="text/csv"
)
