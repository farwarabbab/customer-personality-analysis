# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 13:02:55 2024

@author: Farwa-PC
"""

import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data for example with 23 features
X, Y = make_classification(n_samples=1000, n_features=23, n_classes=2, random_state=42)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Instantiate a Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Fit the model using the training data
rf_model.fit(X_train, Y_train)

# Define function to predict the cluster for the input features
def predict_cluster(data):
    # Predict the cluster using the loaded model
    cluster = rf_model.predict(data.reshape(1, -1))
    return cluster[0]

# Define Streamlit app
def main():
    st.title('Cluster Prediction')
    st.write('Enter values for the following features:')
    
    # User input fields
    education = st.slider('Education', 0, 2, 1)
    income = st.slider('Income', 0, 100000, 50000)
    kid_home = st.slider('Kidhome', 0, 10, 0)
    teen_home = st.slider('Teenhome', 0, 10, 0)
    spent = st.slider('Spent', 0, 10000, 5000)
    wines = st.slider('Wines', 0, 1000, 5000)
    fruits = st.slider('Fruits', 0, 1000, 500)
    meat = st.slider('Meat', 0, 1000, 500)
    fish = st.slider('Fish', 0, 1000, 50)
    sweets = st.slider('Sweets', 0, 1000, 500)
    gold = st.slider('Gold', 0, 1000, 500)
    num_web_purchases = st.slider('NumWebPurchases', 0, 20, 10)
    num_catalog_purchases = st.slider('NumCatalogPurchases', 0, 20, 10)
    num_store_purchases = st.slider('NumStorePurchases', 0, 20, 10)
    num_web_visits_month = st.slider('NumWebVisitsMonth', 0, 50, 25)
    recency = st.slider('Recency', 0, 365, 180)
    age = st.slider('Age', 18, 100, 40)
    complain = st.slider('Complain', 0, 1, 0)
    living_with = st.slider('Living With', 0, 1, 0)
    children = st.slider('Children', 0, 10, 0)
    family_size = st.slider('Family Size', 1, 10, 2)
    is_parent = st.slider('Is Parent', 0, 1, 0)
    customer_for = st.slider('Customer For', 0, 20, 10)

    # Predict button
    if st.button('Predict'):
        # Predict cluster
        cluster = predict_cluster(np.array([education, income, kid_home, teen_home, spent, wines, fruits, meat,
                                             fish, sweets, gold, num_web_purchases, num_catalog_purchases,
                                             num_store_purchases, num_web_visits_month, recency, age,
                                             complain, living_with, children, family_size, is_parent,
                                             customer_for]))
        
        # Display predicted cluster
        st.write('Predicted Cluster:', cluster)

if __name__ == '__main__':
    main()