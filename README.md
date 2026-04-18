# MLOps Churn Prediction Pipeline

An end-to-end MLOps pipeline for customer churn prediction with experiment tracking, model serving, and drift monitoring.

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **scikit-learn** | Model training (RandomForest, LogisticRegression) |
| **MLflow** | Experiment tracking + model registry |
| **FastAPI** | REST API for model serving |
| **Evidently AI** | Data drift monitoring |

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

## 📊 Results

- RandomForest: F1=0.21, ROC-AUC=0.63
- LogisticRegression: F1=0.15, ROC-AUC=0.73
- Drift detected in 2/9 features in production data

## 👤 Author

Karthik Mudenahalli Ashoka
MS Applied AI, Stevens Institute of Technology
