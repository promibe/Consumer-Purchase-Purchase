import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
from artifacts import customergenda_list, productcat, productbrand, predict_purchaseintent
from FeatureEng import FeatureEngineering

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Set page configuration
st.set_page_config(page_title="Consumer Electronic Purchase Intent Prediction", layout="centered")

def create_GUI():
    # Header Section
    st.title("Electronic Purchase Intent Predictor")
    st.write("Predict if a consumer will Purchase a Product or not.")

    # Input Section
    st.header("Consumer Product Features")

    ProductCategory = st.selectbox("Product Category Type", productcat)
    ProductBrand = st.selectbox("Product Brand Type", productbrand)
    ProductPrice = st.text_input("Product Price", "Above 0")
    CustomerAge = st.text_input("Customer Age", "Above 10")
    CustomerGender =st.selectbox("Customer Gender", customergenda_list)
    PurchaseFrequency = st.text_input("Purchase Frequency", "Above 0")
    CustomerSatisfaction = st.text_input("Customer Satisfaction", "Above 0")

    if CustomerGender == 'Male':
        CustomerGender = 0
    else:
        CustomerGender = 1

    # Prediction Section
    st.header("Predicted Purchase Intent")
    if st.button("Predict"):

        try:
            # Ensure price is a float and above 0)
            try:
                ProductPrice = float(ProductPrice)
                if not (ProductPrice > 0):
                    st.error("Price should be above 0")
            except ValueError:
                st.error("Price should be above 0")
                return


            # try to convert the to an integer)
            try:
                CustomerAge = int(CustomerAge)
                if not (CustomerAge > 10):
                    st.error("Age should be above 10")
            except ValueError:
                st.error("Age should be an integer and above 10")
                return


            # try to convert the Purchase frequency to an integer)
            try:
                PurchaseFrequency = int(PurchaseFrequency)
                if not (PurchaseFrequency > 0):
                    st.error("Purchase Frequency should be above 0")
            except ValueError:
                st.error("Purchase Frequency should be an integer and above 10")
                return

            # try to convert the CustomerSatisfaction to an integer)
            try:
                CustomerSatisfaction = int(CustomerSatisfaction)
                if not (CustomerSatisfaction > 0):
                    st.error("Customer Satisfaction should be above 0")
            except ValueError:
                st.error("Customer Satisfaction should be an integer and above 10")
                return

            # If all inputs are valid, create the input data DataFrame

            input_data = pd.DataFrame({
                'ProductCategory':[ProductCategory],
                'ProductBrand':[ProductBrand],
                'ProductPrice':[ProductPrice],
                'CustomerAge':[CustomerAge],
                'CustomerGender':[CustomerGender],
                'PurchaseFrequency':[PurchaseFrequency],
                'CustomerSatisfaction':[CustomerSatisfaction]

            })

            # Call the prediction model
            predicted_intent = predict_purchaseintent(input_data)

            if predicted_intent == 1:
                predicted_intent = 'Have Purchase Intention'
            else:
                predicted_intent = 'Does not have Purchase Intention'

            # Make sure the function is defined correctly
            st.success(f"The Customer {predicted_intent} for {ProductCategory} Product")

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

    else:
        st.info("Enter details and click Predict to see the Purchase Intent.")






create_GUI()

