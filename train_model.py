# train_model.py
# This script trains a Random Forest model on the credit card fraud detection dataset
# and saves the trained model to a file.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv("creditcard.csv")

# Select features
# Note: In a real-world scenario, you would want to preprocess the data (e.g., scaling, handling missing values)
# For this example, we will use the raw features as they are already normalized
features = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4']
X = df[features]
y = df['Class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
model.fit(X_train, y_train)


# Save the trained model to a file
joblib.dump(model, 'model.joblib')
