import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "train_FD001.csv"
MODEL_PATH = BASE_DIR / "models" / "rf_rul_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_columns.pkl"

# ── Thresholds ─────────────────────────────────────────────────────────────────
# Defined once here — used in both the status logic and the trajectory plot.
# Change these values and both places update automatically.
CRITICAL_THRESHOLD = 10
WARNING_THRESHOLD = 40
RUL_TRAINING_CAP = 125    # must match RUL_CAP in preprocessor.py

# ── Sensor config ──────────────────────────────────────────────────────────────
TREND_SENSORS = ["sensor_7", "sensor_9", "sensor_12"]
IMPORTANT_SENSORS = [
    "sensor_7", "sensor_9", "sensor_12",
    "sensor_4", "sensor_11", "sensor_14", "sensor_15"
]

# ── Page config (must be the first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance — RUL",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame | None:
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


# ── Load & validate ────────────────────────────────────────────────────────────
df = load_data()
model, feature_cols = load_model()

if df is None:
    st.error("Dataset not found. Make sure `data/train_FD001.csv` exists.")
    st.stop()

if model is None:
    st.error(
        "Model not found. Run `python -m src.train` first to generate model files.")
    st.stop()

missing_features = [c for c in feature_cols if c not in df.columns]
if missing_features:
    st.error(f"Feature columns missing from dataset: {missing_features}")
    st.stop()

# Only keep sensors that actually exist in the loaded dataset
display_sensors = [s for s in IMPORTANT_SENSORS if s in df.columns]


# ── Title ──────────────────────────────────────────────────────────────────────
st.title("🔧 Predictive Maintenance — RUL Estimator")
st.caption(
    "Monitor engine health and predict remaining useful life from sensor data.")


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Engine Selection")

engine_ids = sorted(df["engine_id"].unique())
engine_id = st.sidebar.selectbox("Select Engine ID", engine_ids)

engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")
min_cycle = int(engine_df["cycle"].min())
max_cycle = int(engine_df["cycle"].max())

# Default the slider to 75% through the engine's life so the first view
# isn't always the failure point (RUL = 0).
default_cycle = min_cycle + int((max_cycle - min_cycle) * 0.75)
selected_cycle = st.sidebar.slider(
    "Inspect at Cycle",
    min_value=min_cycle,
    max_value=max_cycle,
    value=default_cycle,
    help="Move the slider to inspect predicted RUL and health status at any cycle.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Engine **{engine_id}** ran for **{max_cycle} cycles** total.  \n"
    f"Inspecting cycle **{selected_cycle}** of {max_cycle}."
)


# ── Prediction for selected cycle ──────────────────────────────────────────────
selected_row = engine_df[engine_df["cycle"] == selected_cycle]

try:
    X_selected = selected_row[feature_cols]
    predicted_rul = int(model.predict(X_selected)[0])
except Exception as e:
    st.error(f"Prediction failed for cycle {selected_cycle}: {e}")
    st.stop()

actual_rul = (
    int(selected_row["RUL"].values[0])
    if "RUL" in selected_row.columns
    else None
)


# ── Status classification ──────────────────────────────────────────────────────
if predicted_rul <= CRITICAL_THRESHOLD:
    status = "CRITICAL"
    status_color = "error"
elif predicted_rul <= WARNING_THRESHOLD:
    status = "WARNING"
    status_color = "warning"
else:
    status = "HEALTHY"
    status_color = "success"


# ── Metrics row ────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Predicted RUL", f"{predicted_rul} cycles")

with col2:
    if actual_rul is not None:
        delta = predicted_rul - actual_rul
        st.metric(
            "Actual RUL",
            f"{actual_rul} cycles",
            delta=f"{delta:+d} cycles",
            delta_color="inverse",
        )
    else:
        st.metric("Actual RUL", "N/A")

with col3:
    st.metric("Selected Cycle", f"{selected_cycle} / {max_cycle}")

with col4:
    st.metric("Engine Status", status)

# Status alert banner
if status_color == "error":
    st.error(
        f"🚨 Engine **{engine_id}** is in **CRITICAL** condition at cycle {selected_cycle} — "
        f"maintenance required immediately (predicted RUL: {predicted_rul} cycles)."
    )
elif status_color == "warning":
    st.warning(
        f"⚠️ Engine **{engine_id}** is showing **WARNING** signs at cycle {selected_cycle} — "
        f"schedule maintenance soon (predicted RUL: {predicted_rul} cycles)."
    )
else:
    st.success(
        f"✅ Engine **{engine_id}** is **HEALTHY** at cycle {selected_cycle} — "
        f"no immediate action required (predicted RUL: {predicted_rul} cycles)."
    )

st.divider()


# ── RUL Trajectory ─────────────────────────────────────────────────────────────
st.subheader("📈 RUL Prediction Trajectory")

# Predict across every cycle for this engine (full trajectory)
engine_df = engine_df.copy()
engine_df["predicted_rul"] = model.predict(engine_df[feature_cols])

fig_traj, ax_traj = plt.subplots(figsize=(10, 4))

ax_traj.plot(
    engine_df["cycle"], engine_df["predicted_rul"],
    label="Predicted RUL", linewidth=2, color="#1f77b4",
)

if "RUL" in engine_df.columns:
    ax_traj.plot(
        engine_df["cycle"], engine_df["RUL"],
        label="Actual RUL (uncapped)", linewidth=1.5,
        linestyle="--", color="#ff7f0e", alpha=0.8,
    )

# Mark the selected cycle
ax_traj.axvline(
    x=selected_cycle, color="red", linestyle=":",
    linewidth=1.5, label=f"Selected cycle ({selected_cycle})",
)

# Threshold lines — use the named constants, not magic numbers
ax_traj.axhline(
    y=WARNING_THRESHOLD, color="orange", linestyle="--",
    linewidth=1, alpha=0.6, label=f"Warning threshold ({WARNING_THRESHOLD})",
)
ax_traj.axhline(
    y=CRITICAL_THRESHOLD, color="red", linestyle="--",
    linewidth=1, alpha=0.6, label=f"Critical threshold ({CRITICAL_THRESHOLD})",
)

# Training RUL cap line — explains why the actual RUL line spikes above predictions
ax_traj.axhline(
    y=RUL_TRAINING_CAP, color="gray", linestyle=":",
    linewidth=1, alpha=0.5,
    label=f"Training RUL cap ({RUL_TRAINING_CAP}) — model ceiling",
)

ax_traj.set_xlabel("Cycle")
ax_traj.set_ylabel("RUL (cycles)")
ax_traj.set_title(f"RUL Trajectory — Engine {engine_id}")
ax_traj.legend(fontsize=8)
ax_traj.grid(True, alpha=0.3)
plt.tight_layout()

st.pyplot(fig_traj)
plt.close(fig_traj)

st.caption(
    "ℹ️ The model was trained with RUL capped at 125 cycles. "
    "At early cycles the actual RUL (dashed) rises above the cap — "
    "this is expected and does not indicate a model error."
)

st.divider()


# ── Sensor Degradation Trend ───────────────────────────────────────────────────
st.subheader("📉 Sensor Degradation Trend")

sensor_to_plot = st.selectbox("Select Sensor", display_sensors)

fig_sensor, ax_sensor = plt.subplots(figsize=(8, 4))
ax_sensor.plot(engine_df["cycle"], engine_df[sensor_to_plot], linewidth=1.5)
ax_sensor.axvline(
    x=selected_cycle, color="red", linestyle=":",
    linewidth=1.5, label=f"Selected cycle ({selected_cycle})",
)
ax_sensor.set_xlabel("Cycle")
ax_sensor.set_ylabel(sensor_to_plot)
ax_sensor.set_title(f"{sensor_to_plot} Trend — Engine {engine_id}")
ax_sensor.legend(fontsize=8)
ax_sensor.grid(True, alpha=0.3)
plt.tight_layout()

st.pyplot(fig_sensor)
plt.close(fig_sensor)


# ── Sensor Insight ────────────────────────────────────────────────────────────
st.markdown("### 🔍 Sensor Insight")

if sensor_to_plot in TREND_SENSORS:
    st.info(
        f"**{sensor_to_plot}** shows a clear degradation pattern as the engine approaches "
        f"failure. It is strongly correlated with RUL and is a reliable health indicator."
    )
elif sensor_to_plot == "sensor_4":
    st.warning(
        "**sensor_4** is highly important to the model but does not follow a smooth "
        "monotonic degradation trend. The model exploits it through non-linear thresholds "
        "and feature interactions rather than direct time-based changes."
    )
else:
    st.info(
        f"**{sensor_to_plot}** provides contextual information used by the model. "
        f"Its relationship with RUL may be non-linear or conditional on other sensors."
    )

st.divider()


# ── Sensor Readings at Selected Cycle ─────────────────────────────────────────
st.subheader("📄 Sensor Readings at Selected Cycle")

sensor_display_cols = [s for s in display_sensors if s in selected_row.columns]
latest_sensors = selected_row[sensor_display_cols].T.reset_index()
latest_sensors.columns = ["Sensor", "Value"]
latest_sensors["Value"] = latest_sensors["Value"].round(4)

# Include predicted RUL in the table so the download is self-contained
latest_sensors["Predicted RUL"] = predicted_rul

st.dataframe(latest_sensors, use_container_width=True, hide_index=True)


# ── Download ───────────────────────────────────────────────────────────────────
csv = latest_sensors.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Sensor Values + Predicted RUL (CSV)",
    data=csv,
    file_name=f"engine_{engine_id}_cycle_{selected_cycle}_sensors.csv",
    mime="text/csv",
)
