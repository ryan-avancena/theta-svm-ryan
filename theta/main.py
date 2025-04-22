from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

class TrafficData(BaseModel):
    PM_HOUR: float
    BACK_PEAK_HOUR: float
    AHEAD_PEAK_HOUR: float
    BACK_AADT: float
    AHEAD_AADT: float

app = FastAPI()
model = joblib.load("traffic_svm_model.pkl")

@app.post("/predict/")
async def predict_traffic(data: TrafficData):
    try:
        new_data = np.array([[data.PM_HOUR, data.BACK_PEAK_HOUR, data.AHEAD_PEAK_HOUR, data.BACK_AADT, data.AHEAD_AADT]])
        print("📦 Input Array:", new_data)

        prediction = model.predict(new_data)
        print("✅ Prediction Result:", prediction)

        return {"Predicted Traffic Multiplier": round(float(prediction[0]), 2)}
    except Exception as e:
        import traceback
        traceback.print_exc()  # 👈 Print the full error stack trace
        return {"error": str(e)}


@app.get("/")
async def root():
    return {"message": "THETA SVM prediction API is running."}
