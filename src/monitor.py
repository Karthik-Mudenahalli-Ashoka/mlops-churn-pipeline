import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset

# Load original training data as reference
reference_data = pd.read_csv('data/churn.csv')

# Simulate new production data with slight drift
np.random.seed(123)
n = 200
production_data = pd.DataFrame({
    'tenure': np.random.randint(1, 72, n),
    'monthly_charges': np.random.uniform(40, 140, n),
    'total_charges': np.random.uniform(100, 8000, n),
    'contract_type': np.random.choice([0, 1, 2], n),
    'payment_method': np.random.choice([0, 1, 2, 3], n),
    'tech_support': np.random.choice([0, 1], n),
    'online_security': np.random.choice([0, 1], n),
    'num_services': np.random.randint(1, 8, n),
    'churn': np.random.choice([0, 1], n, p=[0.65, 0.35])
})

# Generate drift report
report = Report([DataDriftPreset()])
result = report.run(reference_data=reference_data, current_data=production_data)
result.save_html('data_drift_report.html')

print("✅ Drift report saved to data_drift_report.html")
print("Open data_drift_report.html in your browser!")