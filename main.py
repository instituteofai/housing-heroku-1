from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import pandas as pd

from process_input_data import get_prepared_data

app = FastAPI()

# vectorizer
test_vectorizer = open('model/X_test_vectorized.pkl', 'rb')
test_cv = joblib.load(test_vectorizer)

# model
housing_model = open('model/final_model_aems_housing.pkl', 'rb')
housing_reg = joblib.load(housing_model)

templates = Jinja2Templates(directory='templates')

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

class HousePredictionPayload(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: int
    total_rooms: int
    total_bedrooms: int
    population: int
    households: int
    median_income: float
    ocean_proximity: str



@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(name='index.html', context={
        'request': request,
        'name': 'Chandan Kumar'
    })
@app.post("/")
def process(request: Request):
    return {}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/items/")
def create_item(item: Item):
    return item

@app.get('/predict/')
def predict():
    prediction = housing_reg.predict(test_cv)
    resTop10 = prediction.tolist()[0:10]
    if(len(resTop10) > 0):
        return {'predicted_values': resTop10}
    else:
        return {}

@app.get("/test_results")
def test_results(request: Request):
    return templates.TemplateResponse(name='test_results.html', context={
        'request': request,
        'test_results': predict()
    })

@app.post("/predict/")
def predict_for_one(request: Request, input: HousePredictionPayload):
    # Create pandas dataframe
    payload = {
        "longitude": input.longitude,
        "latitude": input.latitude,
        "housing_median_age": input.housing_median_age,
        "total_rooms": input.total_rooms,
        "total_bedrooms": input.total_bedrooms,
        "population": input.population,
        "households": input.households,
        "median_income": input.median_income,
        "ocean_proximity": input.ocean_proximity
       }
    payload = pd.DataFrame([payload])
    resp = get_prediction(payload)
    return {
        'result': resp.tolist()
    }

# Predict for one value
def get_prediction(payload):
    data_prepared = get_prepared_data(payload)
    prediction_data = housing_reg.predict(data_prepared)
    return prediction_data
    