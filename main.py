from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('model.joblib')

class Transaction(BaseModel):
    time: float
    amount: float
    v1: float
    v2: float
    v3: float
    v4: float

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
    return {"fraud": bool(prediction[0]), "probability": float(model.predict_proba([features])[0][1])}
