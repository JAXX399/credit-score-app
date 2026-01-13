import pandas as pd
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Locate Data
possible_paths = ['german_credit_data.csv', 'data/german_credit_data.csv']
data_path = next((p for p in possible_paths if os.path.exists(p)), None)

if not data_path:
    print("Error: Data not found")
    exit()

# 2. Train Model
df = pd.read_csv(data_path)
target_col = 'credit_risk' if 'credit_risk' in df.columns else df.columns[-1]

X = df.drop([target_col], axis=1)
y = df[target_col]
X_encoded = pd.get_dummies(X)
model_columns = list(X_encoded.columns)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Calculate Class Weight
num_pos = y_train.sum()
num_neg = len(y_train) - num_pos
weight = num_neg / num_pos if num_pos > 0 else 1.0

# XGBoost Classifier
model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    scale_pos_weight=weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train_scaled, y_train)

# 3. Calculate Aggregated Importance
raw_importances = model.feature_importances_
feature_map = {
    "Checking Status": "checking_status",
    "Duration": "duration",
    "Credit History": "credit_history",
    "Purpose": "purpose",
    "Credit Amount": "credit_amount",
    "Savings Account": "savings_status",
    "Employment Since": "employment",
    "Installment Rate": "installment_rate",
    "Sex & Status": "personal_status",
    "Other Debtors": "other_debtors",
    "Residence Since": "residence_since",
    "Property": "property",
    "Age": "age",
    "Other Installments": "other_payment_plans",
    "Housing": "housing",
    "Existing Credits": "existing_credits",
    "Job": "job",
    "num_dependents": "num_dependents",
    "Telephone": "own_telephone",
    "Foreign Worker": "foreign_worker"
}

print("ATTRIBUTE_IMPORTANCE_START")
for name, prefix in feature_map.items():
    matched_indices = [i for i, col in enumerate(model_columns) if col.startswith(prefix)]
    if matched_indices:
        total = sum(raw_importances[i] for i in matched_indices)
        print(f"{name}|{total:.4f}")
print("ATTRIBUTE_IMPORTANCE_END")
