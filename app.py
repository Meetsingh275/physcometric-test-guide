import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Career Prediction System")

st.title("Career Prediction System")

# Load Model
try:
    with open("career_model.pkl", "rb") as file:
        model = pickle.load(file)

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Input Sliders
logical = st.slider("Logical Score", 0, 100)
numerical = st.slider("Numerical Score", 0, 100)
communication = st.slider("Communication Score", 0, 100)
creativity = st.slider("Creativity Score", 0, 100)
leadership = st.slider("Leadership Score", 0, 100)
technical = st.slider("Technical Interest", 0, 100)
helping = st.slider("Helping Nature", 0, 100)
business = st.slider("Business Interest", 0, 100)
artistic = st.slider("Artistic Interest", 0, 100)
stress = st.slider("Stress Handling", 0, 100)

# Prediction
if st.button("Predict Career"):

    sample = pd.DataFrame(
        [[
            logical,
            numerical,
            communication,
            creativity,
            leadership,
            technical,
            helping,
            business,
            artistic,
            stress
        ]],
        columns=[
            "LogicalScore",
            "NumericalScore",
            "CommunicationScore",
            "CreativityScore",
            "LeadershipScore",
            "TechnicalInterest",
            "HelpingNature",
            "BusinessInterest",
            "ArtisticInterest",
            "StressHandling"
        ]
    )

    try:
        prediction = model.predict(sample)
        st.success(f"Predicted Career: {prediction[0]}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")