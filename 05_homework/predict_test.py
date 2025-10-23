import pickle
import requests
from fastapi import Request
from fastapi.encoders import jsonable_encoder

model_file = 'pipeline_v1.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

client = { "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0}

def predict(client):

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }
    
    return jsonable_encoder(result)

url = 'http://localhost:9696/predict'
response = requests.post(url, json=client).json()
print(response)
if response['churn'] == True:
    print("sending promo email to...")