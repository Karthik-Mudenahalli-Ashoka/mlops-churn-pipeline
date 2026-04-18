import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'tenure': np.random.randint(1, 72, n),
    'monthly_charges': np.random.uniform(20, 120, n),
    'total_charges': np.random.uniform(100, 8000, n),
    'contract_type': np.random.choice([0, 1, 2], n),  # 0=month, 1=1yr, 2=2yr
    'payment_method': np.random.choice([0, 1, 2, 3], n),
    'tech_support': np.random.choice([0, 1], n),
    'online_security': np.random.choice([0, 1], n),
    'num_services': np.random.randint(1, 8, n),
})

# Churn logic — longer tenure and contracts = less churn
churn_prob = (
    0.4
    - 0.005 * data['tenure']
    - 0.1 * data['contract_type']
    + 0.002 * data['monthly_charges']
    - 0.05 * data['tech_support']
)
churn_prob = churn_prob.clip(0.05, 0.95)
data['churn'] = (np.random.random(n) < churn_prob).astype(int)

data.to_csv('data/churn.csv', index=False)
print(f"✅ Dataset created: {len(data)} rows, {data['churn'].mean():.1%} churn rate")