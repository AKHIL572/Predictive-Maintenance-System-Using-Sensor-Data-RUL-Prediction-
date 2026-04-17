# 🚀 Predictive Maintenance — Remaining Useful Life (RUL) Prediction




## 📌 Overview

This project focuses on **Predictive Maintenance for Manufacturing Systems**, where the goal is to estimate the **Remaining Useful Life (RUL)** of machines using sensor data.

By predicting when a machine is likely to fail, businesses can:
- ✅ Reduce unplanned downtime
- ✅ Optimize maintenance schedules
- ✅ Improve operational efficiency

---

## 🎯 Problem Statement

In industrial environments, unexpected machine failures can lead to significant financial losses.

> **How can we predict machine failure in advance using sensor data?**

This is modeled as a **Remaining Useful Life (RUL) Regression Problem**, where the model learns degradation patterns from historical sensor readings to estimate how many cycles remain before failure.

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | NASA Turbofan Engine Degradation Dataset |
| **File** | `train_FD001.csv` |
| **Records** | 20,631 rows |

**Features include:**
- Engine ID
- Cycle (time step)
- 3 Operational Settings
- 21 Sensor Readings (`sensor_1` to `sensor_21`)

---

## 🧠 Methodology

### 1️⃣ Data Understanding
- Converted raw `.txt` data into structured `.csv`
- Assigned meaningful column names
- Verified no missing values, no duplicates, and correct data types

### 2️⃣ Exploratory Data Analysis (EDA)
- Engine life cycle distribution
- Sensor trend analysis over time
- Sensor variance analysis
- Correlation heatmap
- Outlier visualization using boxplots

> **Key Insight:** Sensors like **sensor_7** and **sensor_12** show strong degradation patterns correlated with machine failure.

### 3️⃣ Feature Engineering

**RUL Target Variable:**
```python
RUL = max_cycle - current_cycle
```

**Removed low-variance sensors** (near-constant readings that add noise):
`sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`

### 4️⃣ Model Building

| Model | Notes |
|-------|-------|
| Linear Regression | Baseline |
| Ridge Regression | L2 regularization |
| Lasso Regression | L1 regularization |
| **Random Forest Regressor** | ✅ **Best Performer** |

### 5️⃣ Model Evaluation

Metrics used:
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R² Score** — Coefficient of Determination

### 6️⃣ Model Optimization
- Hyperparameter tuning via **GridSearchCV**
- Cross-validation using **GroupKFold** (engine-wise split to prevent data leakage)

### 7️⃣ Deployment

Built an interactive dashboard with **Streamlit**:
- 🔍 Select Engine ID
- ⏱️ **Cycle Slider:** Inspect engine health at any specific point in its lifetime
- ⚡ Predict RUL instantly for the selected cycle
- 📈 **RUL Trajectory:** Visualize the engine's entire simulated health trajectory over time against critical thresholds
- 📉 Visualize individual sensor degradation trends
- 📋 View sensor values for the selected cycle
- 💾 Download results as CSV

---

## 🏗️ Project Structure

```
PREDICTIVE_MAINTENANCE/
│
├── data/
│   └── train_FD001.csv
│
├── models/
│   ├── rf_rul_model.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   ├── 1_data_understanding.ipynb
│   ├── 2_eda.ipynb
│   └── 3_preprocessing_&_modeling.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── train.py
│   └── predict.py
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/predictive-maintenance.git
cd predictive-maintenance
```

### 2. Create a Virtual Environment
```bash
python -m env_name

# Windows
venv\Scripts\activate

# macOS / Linux
source env_name/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app/app.py
```

---

## 📈 Sample Output

```
Engine ID     : 42
Predicted RUL : 6 cycles
Engine Status : 🔴 CRITICAL
```

The dashboard also renders dynamic sensor trend visualizations for the selected engine.

---

## 💡 Key Learnings

- Time-series degradation modeling for industrial machinery
- Feature selection using variance thresholding
- Handling grouped data with engine-wise train/validation splits
- Building production-ready ML pipelines with Scikit-learn
- Deploying ML models as interactive apps using Streamlit

---

## 🚀 Future Improvements

- [ ] Implement **LSTM** for sequence-based RUL prediction
- [ ] Add **real-time data streaming** support
- [ ] Deploy on cloud (**AWS / Azure / Streamlit Cloud**)
- [ ] Build a **REST API** using FastAPI
- [ ] Extend to multi-condition datasets (FD002, FD003, FD004)

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib |
| Deployment | Streamlit |

---

## 👨‍💻 Author

**Akhil T V**  
*Aspiring Data Scientist*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/akhil-t-v/)

---