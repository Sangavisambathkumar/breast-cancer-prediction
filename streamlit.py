import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier

# Load and preprocess data
df = pd.read_csv('breast_cancer_data.csv')
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_scaled, y)

# Load trained model
model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam')
model.fit(X_selected, y)

# Streamlit app
st.title("Breast Cancer Prediction App")
st.write("Interactive Web App for Breast Cancer Analysis")

user_input = st.text_input("Enter comma-separated feature values:")
if user_input:
    input_data = [float(i) for i in user_input.split(',')]
    scaled_input = scaler.transform([input_data])
    selected_input = selector.transform(scaled_input)
    prediction = model.predict(selected_input)
    st.write("Prediction (1 = Malignant, 0 = Benign):", prediction[0])
