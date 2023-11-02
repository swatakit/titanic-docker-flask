# app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
from utils.CustomTransformers import *
import logging

logging.basicConfig(filename='/app/log/prediction.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

app = Flask(__name__)

# Load the model
model = joblib.load('titanic_model.sav')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        features = request.json

        # if model is a pipeline that built by random search cv
        if hasattr(model, 'estimator'):
            prediction = model.predict_proba(pd.DataFrame([features]))[:,1]
        else:
            prediction = model.transform(pd.DataFrame([features]))[:,1]

        logging.info(f"Prediction made: {prediction}, Features: {features}")

        return jsonify({"prediction": str(prediction)})
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        return jsonify({'error': 'Something went wrong!'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

