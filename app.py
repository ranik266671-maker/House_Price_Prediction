import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("house_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page title
st.title("🏠 House Price Prediction")
st.write("Enter house details to predict its price.")

# User inputs
area = st.number_input(
    "Area (square feet)", min_value=100, value=1500
)

bedrooms = st.number_input(
    "Number of Bedrooms", min_value=1, value=3, step=1
)

bathrooms = st.number_input(
    "Number of Bathrooms", min_value=1, value=2, step=1
)

stories = st.number_input(
    "Number of Stories", min_value=1, value=2, step=1
)

parking = st.number_input(
    "Number of Parking Spaces", min_value=0, value=1, step=1
)

# Prediction button
if st.button("Predict House Price"):

    input_data = pd.DataFrame(
        [[area, bedrooms, bathrooms, stories, parking]],
        columns=[
            "area",
            "bedrooms",
            "bathrooms",
            "stories",
            "parking"
        ]
    )

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted House Price: ₹{prediction:,.0f}")