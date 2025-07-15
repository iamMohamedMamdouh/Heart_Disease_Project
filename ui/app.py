import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

model = joblib.load("D:\Coding\Python\Heart_Disease_Project\models/final_model.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("Heart Disease Risk Predictor")

st.markdown("Please enter the following health information:")

age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise-Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0–4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect, 3 = Other)", [0, 1, 2, 3])

sex = 1 if sex == "Male" else 0

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Result: The patient is at risk of heart disease.")
    else:
        st.success("Result: No signs of heart disease.")

st.markdown("---")
st.subheader("Data Insights")

df = pd.read_csv("D:/Coding/Python/Heart_Disease_Project/heart+disease/heart_disease.csv")

fig1, ax1 = plt.subplots()
sns.boxplot(x='target', y='thalach', data=df, ax=ax1)
ax1.set_title("Max Heart Rate vs Heart Disease")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax2)
ax2.set_title("Correlation Heatmap")
st.pyplot(fig2)
