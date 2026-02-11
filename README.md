# 🔧 Predictive Maintenance System (Remaining Useful Life Prediction)

## 📌 Project Overview
This project implements an **end-to-end Predictive Maintenance system** to estimate the **Remaining Useful Life (RUL)** of industrial machinery using sensor data.  
The goal is to **predict equipment failure in advance** so that maintenance can be scheduled proactively, avoiding unplanned downtime and costly breakdowns.

The system uses historical sensor readings from multiple engines and applies **machine learning models** to forecast how many operational cycles remain before failure.

---

## 🎯 Objectives
- Predict Remaining Useful Life (RUL) of machines using sensor data
- Identify sensor degradation patterns leading to failure
- Compare multiple regression models and select the best-performing one
- Build a **modular, reusable ML pipeline**
- Deploy predictions using an **interactive Streamlit dashboard**

---

## 🏭 Real-World Relevance
Predictive maintenance is widely used in:
- Manufacturing plants
- Aviation (engine health monitoring)
- Power plants
- Heavy machinery and industrial IoT systems

This project mirrors **real industrial workflows**, including:
- Engine-wise data splitting (to avoid data leakage)
- Group-based cross-validation
- Sensor selection based on variance and correlation
- Model explainability through feature importance

---

## 📊 Dataset
**Source:** NASA C-MAPSS Turbofan Engine Degradation Dataset  
**Type:** Public research dataset (not a Kaggle competition)

**Data characteristics:**
- Multiple engines
- Multivariate time-series sensor data
- Failure occurs naturally over time
- RUL calculated from max cycle per engine

---

## 🗂 Project Structure
```
predictive_maintenance/
│
├── dataset/
│ ├── train_FD001.txt
│ └── train_FD001.csv
│
├── models/
│ └── rul_prediction_model.joblib
│
├── notebooks/
│ ├── 1_data_understanding.ipynb
│ ├── 2_eda.ipynb
│ └── 3_preprocessing_&_modeling.ipynb
│
├── src/
│ ├── data_loader.py
│ ├── preprocessor.py
│ ├── train.py
│ └── predict.py
│
├── app.py
├── requirements.txt
```


---

## 🔬 Methodology

### 1️⃣ Data Understanding
- Converted raw `.txt` sensor data into structured CSV format
- Verified data types, missing values, and duplicates
- Identified constant and non-informative sensors

### 2️⃣ Exploratory Data Analysis (EDA)
- Sensor variance analysis
- Sensor correlation heatmaps
- Engine life cycle distribution
- Sensor degradation trends over time
- RUL distribution analysis

### 3️⃣ Feature Engineering
- Calculated Remaining Useful Life (RUL)
- Removed sensors with:
  - Near-zero variance
  - No degradation signal
- Selected informative sensors based on:
  - Variance
  - Correlation
  - Feature importance

### 4️⃣ Model Training
Models evaluated:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor

Best model:
- **Random Forest Regressor**

### 5️⃣ Model Validation
- Engine-wise train-test split
- GroupKFold cross-validation
- Evaluation metrics:
  - MAE
  - RMSE
  - R² Score

---

## 🧪 Model Performance (Final)
| Metric | Value |
|------|------|
| MAE  | ~25.8 |
| RMSE | ~35.2 |
| R²   | ~0.71 |

---

## 📈 Feature Importance
Top sensors contributing to RUL prediction:
- `sensor_9`
- `sensor_7`
- `sensor_12`
- `sensor_4`

These sensors show clear degradation patterns as engines approach failure.

---

## 🖥 Streamlit Application
The project includes an interactive **Streamlit dashboard** that allows:

- Engine selection (1–100)
- Real-time RUL prediction
- Engine health classification:
  - 🟢 Healthy
  - 🟠 Warning
  - 🔴 Critical
- Sensor degradation visualization
- Downloadable latest sensor values as CSV

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

2️⃣ Train the Model
```bash
python -m src.train
```

3️⃣ Run Prediction Script
```bash
python -m src.predict
```

4️⃣ Launch Streamlit App
```bash
python -m streamlit run app.py
