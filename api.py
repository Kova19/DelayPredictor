'''
Bachelor thesis FIT VUT
Author: Martin Kováčik (xkovacm01)
Date: 20.4.2026

Main API file for the delay predictor. This file contains the FastAPI app and the endpoint for predicting delays.
'''

from fastapi import FastAPI
from predict import predictDelay

app = FastAPI()


@app.post("/predict")
def predict(data: dict):
    try:
        if data["visualization"] is True:
            shape, realtime, prediction = predictDelay(data)
            if all(v == -1 for v in (shape, realtime, prediction)):
                return {"code": 501, "error": "Predictor did not found line ID for route"}
            if all(v == -2 for v in (shape, realtime, prediction)):
                return {"code": 502, "error": "Predictor did not found tripID for prection"}
            if all(v == -3 for v in (shape, realtime, prediction)):
                return {"code": 503, "error": "Predictor did not found average delays for trip"}

            return {"shape": shape, "realtime": realtime, "prediction": prediction}
        else:
            prediction = predictDelay(data)
            if prediction == -22:
                return {"error": "Require stop for prediction without vizualization"}
            return {"prediction": prediction}
    except Exception as e:
        print(e)
        return {"error": str(e)}
