import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("titanic_model.pkl")
scaler = joblib.load("titanic_scaler.pkl")

# UI
st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details and click Predict to see if they'd survive.")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["Female", "Male"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouse aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"])

# Preprocess inputs
sex = 0 if sex == "Female" else 1
embarked_map = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}
embarked = embarked_map[embarked]

# Scale age and fare
scaled_values = scaler.transform([[age, fare]])[0]
age_scaled, fare_scaled = scaled_values[0], scaled_values[1]

# Final array
input_data = np.array([[pclass, sex, age_scaled, sibsp, parch, fare_scaled, embarked]])

# Predict
if st.button("Predict"):
    result = model.predict(input_data)[0]
    if result == 1:
        st.success("🎉 Survived!")
    else:
        st.error("❌ Did Not Survive")
