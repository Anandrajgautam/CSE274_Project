# 🧠 Neurological AI Disease Detector

An interactive, machine-learning-powered web application for the early screening and risk assessment of major neurological conditions: **Alzheimer's Disease**, **Parkinson's Disease**, and **Stroke**. 

This project demonstrates a cost-effective healthcare solution using **Semi-Supervised Learning (SSL)**, reducing the need for expensive labeled medical data by up to 80% while maintaining high diagnostic accuracy.

---

## 🌟 Key Features

### 1. 🧠 Alzheimer's Disease Prediction
* **Clinical Assessment:** Uses a comprehensive 32-feature clinical model based on demographics, lifestyle, medical history, cognitive assessment scores (MMSE, ADL), and symptoms.
* **Risk Categorization:** Provides dynamic risk scoring and generates interactive gauges.
* **Actionable Advice:** Displays clinical recommendations categorized by risk level (Low, Moderate, High).

### 2. 🎤 Parkinson's Disease Detection
* **Acoustic Analysis:** Analyzes voice recording uploads (e.g., sustained phonation of 'ahhh...') using **Praat-Parselmouth** and **Librosa**.
* **Feature Extraction:** Extracts 22 voice metrics including pitch, jitter (frequency instability), shimmer (amplitude perturbation), Harmonic-to-Noise Ratio (HNR), and non-linear complexity metrics (RPDE, DFA, PPE).
* **Manual Input Alternative:** Allows manual input of clinical voice features if audio libraries aren't available locally.

### 3. ❤️ Stroke Risk Assessment
* **Vascular & Lifestyle Factors:** Analyzes demographics (age, gender), medical history (hypertension, heart disease), lifestyle (work type, residence, smoking status), and clinical metrics (average glucose level, BMI).
* **Imbalance Handling:** Utilizes SMOTE (Synthetic Minority Over-sampling Technique) to handle severe class imbalances in stroke prevalence.

---

## 🔬 Semi-Supervised Learning (SSL) Paradigm

Acquiring annotated clinical data is expensive, time-consuming, and requires specialized medical expertise. To address this, our model uses a **Self-Training** approach:
* **The Process:** The models are trained initially on a small set of labeled data (20–30%). They then iteratively predict labels for unlabeled data and add the most confident predictions to the training set.
* **Benefits:** 
  * Achieve high classification accuracy comparable to fully supervised methods.
  * Reduce annotation cost and effort by up to **80%**.
  * Use robust classifiers: **XGBoost** for Parkinson's & Stroke, and **Gradient Boosting** for Alzheimer's.

---

## 📊 Model Performance Summary

| Disease / Condition | Base Classifier | Accuracy | Precision | Recall (Sensitivity) | F1-Score | AUC-ROC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **🧠 Alzheimer's** | Gradient Boosting | **92.1%** | **93.4%** | **83.6%** | **88.2%** | **0.933** |
| **🎤 Parkinson's** | XGBoost | **79.5%** | **80.0%** | **96.6%** | **87.5%** | **0.928** |
| **❤️ Stroke** | XGBoost + SMOTE | **84.6%** | **15.5%** | **48.0%** | **23.4%** | **0.799** |

*Note: For Stroke Risk, low precision is normal due to the severe clinical class imbalance (~4.8% prevalence). However, the model successfully captures 48% of high-risk stroke patients during screening.*

---

## 📂 Project Structure

```text
CSE274_Project/
├── Parkinsson disease.csv            # Parkinson's voice acoustics dataset
├── alzheimers_disease_data.csv       # Alzheimer's clinical dataset
├── healthcare-dataset-stroke-data.csv # Stroke risk factor dataset
├── app.py                            # Streamlit web application dashboard
├── train_model.ipynb                 # Jupyter notebook containing training pipeline
├── models/                           # Directory to store trained models (*.pkl) (gitignored)
│   ├── parkinsons_model.pkl
│   ├── alzheimers_model.pkl
│   └── stroke_model.pkl
├── results/                          # Directory containing performance graphs & plots
│   ├── alzheimers_cm.png
│   ├── parkinsons_cm.png
│   ├── stroke_cm.png
│   └── ...                           # Other ROC/feature importance charts
├── .gitignore                        # Standard files, cache, and large folders to ignore
└── README.md                         # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Clone the Repository
```bash
git clone https://github.com/Anandrajgautam/CSE274_Project.git
cd CSE274_Project
```

### 3. Create a Virtual Environment
```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on macOS/Linux:
source .venv/bin/activate
```

### 4. Install Dependencies
Install all the required Python libraries:
```bash
pip install streamlit numpy pandas scikit-learn xgboost imbalanced-learn joblib plotly matplotlib seaborn
```

*Optional for audio extraction feature:*
To enable voice file analysis, install:
```bash
pip install librosa soundfile praat-parselmouth
```

---

## 🚀 How to Run the Project

### Step 1: Train the Models
Ensure the three CSV datasets are placed in the root directory. Run all cells in `train_model.ipynb` to build, evaluate, and save the model pickles into the `models/` directory.

### Step 2: Start the Web Dashboard
Launch the Streamlit app using the command:
```bash
streamlit run app.py
```

### Step 3: Access the App
Open the local server link generated in your terminal (typically `http://localhost:8501`) in your web browser. Use the sidebar/top tabs to toggle between the three neurological detectors.

---

## ⚠️ Disclaimer
This software is a **clinical decision support screening tool** and does not constitute medical advice, diagnosis, or treatment. All predictions should be interpreted by qualified healthcare professionals.
