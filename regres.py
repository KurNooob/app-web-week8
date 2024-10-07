import pandas as pd
import joblib
import streamlit as st

# Load the trained model
loaded_model = joblib.load('life_xp.pkl')

# Streamlit app
st.title("Predict Life Expectancy")

# User inputs for features
bmi = st.slider("Select BMI", min_value=10.0, max_value=35.0, value=25.0, step=0.1)
gdp_per_capita = st.slider("Select GDP per Capita", min_value=0, max_value=120000, value=10000, step=100)
schooling = st.slider("Select Years of Schooling", min_value=0, max_value=20, value=12, step=1)

# Button to predict
if st.button("Predict Life Expectancy"):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'BMI': [bmi],
        'GDP_per_capita': [gdp_per_capita],
        'Schooling': [schooling]
    })

    # Predict life expectancy
    predicted_life_expectancy = loaded_model.predict(input_data)
    st.success(f"Predicted Life Expectancy: {predicted_life_expectancy[0]:.2f} years")