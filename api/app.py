# api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Robust model path handling
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(MODEL_PATH)

# Define the input data model
class Transaction(BaseModel):
    time: float
    amount: float
    v1: float
    v2: float
    v3: float
    v4: float

@app.get("/")
def read_root():
    return {"status": "API is up and running!"}

@app.post("/predict")
async def predict(transaction: Transaction):
    features = [
        transaction.time,
        transaction.amount,
        transaction.v1,
        transaction.v2,
        transaction.v3,
        transaction.v4
    ]
    prediction = model.predict([features])
    probability = float(model.predict_proba([features])[0][1])
    return {
        "fraud": bool(prediction[0]),
        "probability": probability
    }


#use uvicorn to run the app
# To run the app, use the command:  
# uvicorn api.app:app --reload
# This will start the FastAPI server and you can access the API at http://