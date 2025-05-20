# train_model.py
# This script trains a Random Forest model on the credit card fraud detection dataset
# and saves the trained model to api/model.joblib.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Path to your dataset (adjust if needed)
DATA_PATH = os.path.join("data", "creditcard.csv")

# Load data
df = pd.read_csv(DATA_PATH)

# Select features
features = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4']  # Adjust based on your dataset
X = df[features] # Features
y = df['Class'] # Target variable (fraud or not)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y) # Ensure stratified split

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
model.fit(X_train, y_train)

# Ensure the api directory exists
os.makedirs("api", exist_ok=True)

# Save the trained model to the api/ directory
MODEL_PATH = os.path.join("api", "model.joblib")
joblib.dump(model, MODEL_PATH)

print(f"Model trained and saved to {MODEL_PATH}")
