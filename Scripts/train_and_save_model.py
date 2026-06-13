import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Load dataset
data_path = r"C:\Users\vvars\OneDrive\Desktop\customer churn ml\data.csv"
data = pd.read_csv(data_path)

# Data preprocessing
data = data.drop(["customerID"], axis=1)

# Clean TotalCharges and tenure
data['TotalCharges'] = pd.to_numeric(data.TotalCharges, errors='coerce')
data.drop(labels=data[data["tenure"] == 0].index, axis=0, inplace=True)
data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)

# Categorical column mapping
cat_cols = []
label_encoders = {}
for col in data.columns:
    if not pd.api.types.is_numeric_dtype(data[col]):
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = {
            "classes": le.classes_.tolist(),
            "mapping": {val: int(idx) for idx, val in enumerate(le.classes_)}
        }
        cat_cols.append(col)

# Train-test split
X = data.drop(columns="Churn")
y = data["Churn"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4, stratify=y)

# Scale numeric columns
col_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
X_train[col_to_scale] = scaler.fit_transform(X_train[col_to_scale])
X_test[col_to_scale] = scaler.transform(X_test[col_to_scale])

# Train voting classifier
clf1 = GradientBoostingClassifier()
clf2 = LogisticRegression()
clf3 = AdaBoostClassifier()
model = VotingClassifier(estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)], voting='soft')
model.fit(X_train, y_train)

# Save assets
assets = {
    "model": model,
    "scaler": scaler,
    "label_encoders": label_encoders,
    "scale_cols": col_to_scale,
    "feature_order": list(X.columns)
}

output_dir = r"C:\Users\vvars\OneDrive\Desktop\customer churn ml\Scripts"
with open(os.path.join(output_dir, "churn_model_assets.pkl"), "wb") as f:
    pickle.dump(assets, f)

print("Model trained and assets saved successfully!")
