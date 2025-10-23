import pickle
from fastapi import FastAPI
from fastapi import Request
from fastapi.encoders import jsonable_encoder
import uvicorn

model_file = 'pipeline_v1.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI()

@app.post('/predict')

async def predict(request: Request):
    client = await request.json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }
    
    return jsonable_encoder(result)


if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=9696, reload=True)