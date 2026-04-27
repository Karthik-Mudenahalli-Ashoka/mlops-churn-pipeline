# MLOps Churn Prediction Pipeline

An end-to-end MLOps pipeline for customer churn prediction with experiment tracking, model serving, and drift monitoring — built to reflect production-grade ML engineering practices.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| scikit-learn | Model training (RandomForest, LogisticRegression) |
| MLflow | Experiment tracking + model registry |
| FastAPI | REST API for model serving |
| Evidently AI | Data drift monitoring |

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

### Serve API

```bash
uvicorn app:app --reload --port 8000
```

Open http://localhost:8000/docs

### Monitor Drift

```bash
python src/monitor.py
open data_drift_report.html
```

---

## 📊 Results

- RandomForest: F1=0.21, ROC-AUC=0.63
- LogisticRegression: F1=0.15, ROC-AUC=0.73
- Drift detected in 2/9 features in production data

---

## 🏢 Enterprise Application

Building a model is only 20% of the work — the other 80% is making sure it continues to work reliably in production. This pipeline demonstrates the infrastructure layer required for serious AI adoption at scale:

- **MLflow Experiment Tracking** — Every training run is logged with parameters, metrics, and model artifacts, enabling full reproducibility and audit trails. Teams can compare runs, roll back to previous versions, and share results across the organization
- **FastAPI Model Serving** — The trained model is exposed as a REST endpoint, making it instantly consumable by any internal application, dashboard, or downstream service without requiring a data scientist in the loop
- **Evidently AI Drift Monitoring** — Automatically detects when incoming production data no longer matches the training distribution, triggering alerts before model performance silently degrades

> **Why this matters for AI Enablement:** Deploying AI across a large enterprise means you can't just ship a model and walk away. This pipeline represents the operational backbone that keeps AI solutions reliable, explainable, and maintainable at scale — a critical component of any responsible AI adoption strategy across business functions.

**Real-world scenario:** A churn model deployed for a rental or mobility company needs to be retrained as customer behavior shifts seasonally. This pipeline automates that lifecycle — from retraining to redeployment to monitoring — without requiring manual intervention each cycle.

---

## 👤 Author

**Karthik Mudenahalli Ashoka**  
MS in Applied Artificial Intelligence, Stevens Institute of Technology  
[LinkedIn](https://www.linkedin.com/in/m-a-karthik/) | [GitHub](https://github.com/Karthik-Mudenahalli-Ashoka)
