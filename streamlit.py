import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
import numpy as np

# Function to load the dataset and preprocess
def load_data():
    df = pd.read_csv('breast_cancer_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y

# Function to scale the input data
def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled

# Feature selection function
def select_features(X_scaled, y):
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X_scaled, y)
    return selector, X_selected

# Function to load and train the model
def train_model(X_selected, y):
    model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam')
    model.fit(X_selected, y)
    return model

# Streamlit app
st.title("Breast Cancer Prediction App")
st.write("This is an interactive web app for predicting whether breast cancer is malignant or benign.")

# Load and preprocess data
X, y = load_data()
scaler, X_scaled = preprocess_data(X)
selector, X_selected = select_features(X_scaled, y)
model = train_model(X_selected, y)

# Explain the features
st.write("""
### Features:
1. The model requires 10 comma-separated numerical features as input.
2. These features correspond to measurements of the cancerous cells (e.g., radius, texture, smoothness, etc.)
3. The model predicts whether the tumor is **Malignant (1)** or **Benign (0)** based on these features.
""")

# User input for prediction
user_input = st.text_input("Enter 10 comma-separated feature values:")

if user_input:
    try:
        # Convert input to a list of floats
        input_data = [float(i) for i in user_input.split(',')]
        
        # Check if exactly 10 values are entered
        if len(input_data) != 10:
            st.error("Please enter exactly 10 values.")
        else:
            # Scale and select features
            scaled_input = scaler.transform([input_data])
            selected_input = selector.transform(scaled_input)

            # Predict
            prediction = model.predict(selected_input)
            
            # Display the result
            if prediction[0] == 1:
                st.write("Prediction: Malignant (1) - The tumor is cancerous.")
            else:
                st.write("Prediction: Benign (0) - The tumor is not cancerous.")
    except ValueError:
        st.error("Please enter valid numerical values.")

