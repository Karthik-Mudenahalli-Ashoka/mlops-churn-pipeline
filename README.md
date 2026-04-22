# 🔄 MLOps Customer Churn Prediction Pipeline

An end-to-end MLOps pipeline for predicting customer churn — from data ingestion to model deployment and real-time drift monitoring.

🚀 **Live API:** [localhost:8000/docs](http://localhost:8000/docs) (run locally)

---

## 🎯 Business Problem

Customer churn is one of the costliest challenges in financial services and insurance. Losing a customer means lost premiums, increased acquisition costs, and reduced lifetime value. This pipeline helps businesses **identify at-risk customers before they leave**, enabling proactive retention strategies that protect revenue.

**Real-world applications at insurance companies:**
- Identify policyholders likely to cancel their coverage
- Flag customers for targeted retention outreach
- Monitor model health over time as customer behavior shifts

---

## 🧠 How It Works

```
Raw Customer Data
      ↓
Feature Engineering & Preprocessing
      ↓
Model Training (RandomForest + LogisticRegression)
      ↓
Experiment Tracking with MLflow
      ↓
Best Model Registered & Served via FastAPI
      ↓
Real-time Data Drift Monitoring with Evidently AI
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **scikit-learn** | Model training (RandomForest, LogisticRegression) |
| **MLflow** | Experiment tracking + model registry + versioning |
| **FastAPI** | REST API for real-time churn predictions |
| **Evidently AI** | Production data drift monitoring |
| **Docker** | Containerized deployment |

---

## 📊 Business Impact

| Metric | Value |
|---|---|
| Best ROC-AUC | 0.73 (LogisticRegression) |
| Drift Detection | 2 out of 9 features flagged in production |
| Deployment | REST API serving predictions in real time |
| Model Versioning | Full experiment history tracked via MLflow |

> The drift detection capability is critical for production ML systems — it ensures the model stays accurate as customer behavior changes over time, a key requirement in regulated industries like insurance.

---

## ✨ Key Features

- 🔁 **End-to-end pipeline** — data → training → serving → monitoring
- 📊 **Experiment tracking** — compare models, log metrics, register best model
- ⚡ **REST API** — real-time churn predictions via FastAPI
- 📉 **Drift monitoring** — automated alerts when production data shifts
- 🐳 **Dockerized** — portable and production-ready

---

## 🚀 Getting Started

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Generate Data
```bash
python src/generate_data.py
```

### Train Models
```bash
python src/train.py
```

### View MLflow Dashboard
```bash
mlflow ui
```
Open http://localhost:5000

### Serve Prediction API
```bash
uvicorn app:app --reload --port 8000
```
Open http://localhost:8000/docs

### Monitor Data Drift
```bash
python src/monitor.py
open data_drift_report.html
```

---

## 📁 Project Structure

```
mlops-churn-pipeline/
│
├── src/
│   ├── generate_data.py     # Synthetic customer data generation
│   ├── train.py             # Model training + MLflow logging
│   └── monitor.py           # Evidently drift monitoring
├── models/                  # Saved model artifacts
├── data/                    # Raw and processed data
├── app.py                   # FastAPI prediction endpoint
├── Dockerfile               # Container configuration
├── requirements.txt         # Dependencies
└── README.md
```

---

## 🔮 Future Improvements

- Integration with Azure ML for cloud-based training and deployment
- Power BI dashboard for business stakeholders to view churn risk scores
- SHAP explainability — show *why* a customer is flagged as at-risk
- Automated retraining pipeline when drift is detected
- Support for real customer datasets (telecom, insurance, banking)

---

## 👤 Author

**Karthik Mudenahalli Ashoka**
- MS in Applied Artificial Intelligence — Data Engineering Concentration
- Stevens Institute of Technology
- [LinkedIn](https://www.linkedin.com/in/m-a-karthik/) | [GitHub](https://github.com/Karthik-Mudenahalli-Ashoka)
