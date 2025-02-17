from fastapi import FastAPI
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.losses import MeanSquaredError

app = FastAPI()
model = load_model("traffic_cnn_model.h5", compile=False)  # Load model without compiling
model.compile(optimizer="adam", loss=MeanSquaredError(), metrics=["mae"])  # Explicitly define loss function

@app.post("/predict/")
async def predict_traffic(data: dict):
    new_data = np.array([[data["PM_HOUR"], data["BACK_PEAK_HOUR"], data["AHEAD_PEAK_HOUR"], data["BACK_AADT"], data["AHEAD_AADT"]]])
    new_data = np.expand_dims(new_data, axis=-1)
    prediction = model.predict(new_data)
    return {"Predicted Traffic Volume": float(prediction[0][0])}

# Run the API
# uvicorn filename:app --reload
