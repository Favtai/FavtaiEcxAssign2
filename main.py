import fastapi
print(fastapi.__version__)


from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

app = FastAPI()


class Entry(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    redisual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def read_root():
    return {"Project": "Wine Prediction Project."}


@app.post("/prediction")
def make_prediction(entry: Entry):
    """Takes in a request with the features of a wine and returns the predicted quality of the wine."""
    
    data = pd.read_csv('./data/winequality-red.csv')
    data.drop('quality', axis=1, inplace=True)

    model = pickle.load(open('./model/wine_rmodel.pkl', 'rb'))
    
    # test_df = pd.DataFrame([entry.dict().values()], columns=data.columns)
    print(list(entry.dict().values()))
    # print()
    # return {"result": model.predict(test_df)}
    return model.predict([np.array(list(entry.dict().values()))])