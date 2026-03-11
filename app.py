from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("crop_yield_model.pkl")

@app.get("/")
def home():
    return {"message": "Crop Yield Prediction API"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # apply same encoding
    df = pd.get_dummies(df)

    # load training columns
    model_columns = model.feature_names_in_

    # align columns
    df = df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df)

    return {"prediction": float(prediction[0])}