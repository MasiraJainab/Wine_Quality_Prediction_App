import numpy as np
import joblib

app_code = '''
import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("wine_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.set_page_config(page_title="Wine Quality Predictor", layout="centered")
st.title("🍷 Wine Quality Prediction App")
st.write("Enter the chemical properties of wine to predict if it's **Good** or **Not Good**.")

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, step=0.0001, format="%.4f")
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=1.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=1.0)
density = st.number_input("Density", min_value=0.0, step=0.0001, format="%.4f")
pH = st.number_input("pH", min_value=0.0, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)

# Prediction
if st.button("Predict Wine Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                            residual_sugar, chlorides, free_sulfur_dioxide,
                            total_sulfur_dioxide, density, pH, sulphates, alcohol]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    label = "🟢 Good Quality Wine" if prediction == 1 else "🔴 Not Good Quality Wine"
    st.subheader(f"Prediction: {label}")
'''

with open("app.py", "w") as f:
    f.write(app_code)

requirements_code = '''
streamlit
scikit-learn
joblib
numpy
'''

with open("requirements.txt", "w") as f:
    f.write(requirements_code.strip())

print("app.py and requirements.txt created!")
