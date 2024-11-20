import cloudpickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from FeatureEng import FeatureEngineering

with open('ProductCategory_data.txt', 'r') as file:
    productcat = [ line.strip().strip("'") for line in file.readlines()]


with open('ProductBrand_data.txt', 'r') as file:
    productbrand = [ line.strip().strip("'") for line in file.readlines()]

customergenda_list = ['Male', 'Female']


try:
    with open('Con_Elec_pipeline.joblib', 'rb') as file:
        model = cloudpickle.load(file)
        print(model)
except Exception as e:
    print(f"Error while loading model: {str(e)}")


def predict_purchaseintent(df):
    #perform the prediction
    result = model.predict(df)
    if result is not None and isinstance(result, np.ndarray):  # converting it to a scaler
        result = result[0]
    return result