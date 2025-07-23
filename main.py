from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import os

app = FastAPI()

# Corrected file
model_path = os.path.join(os.path.dirname(_file_), "model.pkl")
model = joblib.load(model_path)

# Corrected class name and typing
class Student(BaseModel):
    features: List[float]

@app.post("/predict")
def predict(data: Student):
    prediction = model.predict([data.features])
    return {"prediction": "Pass" if prediction[0] == 1 else "Fail"}