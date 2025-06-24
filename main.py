from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model
model = joblib.load("model.joblib")

# Define FastAPI app
app = FastAPI()

# Define input schema
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

# Prediction endpoint
@app.post("/predict")
def predict_survival(data: Passenger):
    input_df = pd.DataFrame([data.dict()])

    # Encode categorical features
    input_df["Sex"] = input_df["Sex"].replace({"male": 0, "female": 1})
    input_df["Embarked"] = input_df["Embarked"].replace({'S': 0, 'C': 1, 'Q': 2})

    # Predict
    prediction = model.predict(input_df)[0]
    return {"Survived": bool(prediction)}