import pickle
import requests
from flask import request
from flask import jsonify

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

client = {"job": "management", "duration": 400, "poutcome": "success"}

def predict(client):

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }
    
    return jsonify(result)

url = 'http://localhost:9696/predict'
response = requests.post(url, json=client).json()
print(response)
if response['churn'] == True:
    print("sending promo email to...")