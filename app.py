import streamlit as st
import pandas as pd
import pickle

# Load model
with open("career_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Career Prediction System")

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

if st.button("Predict Career"):

    sample = pd.DataFrame([[logical, numerical, communication,
                            creativity, leadership, technical,
                            helping, business, artistic, stress]],
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
    ])

    prediction = model.predict(sample)

    st.success(f"Predicted Career: {prediction[0]}")