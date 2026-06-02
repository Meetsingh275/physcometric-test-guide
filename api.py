from fastapi import FastAPI
import pandas as pd
import pickle

app = FastAPI()

with open("career_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.get("/")
def home():
    return {"message": "Career Prediction API Running"}

@app.get("/predict")
def predict(
    logical: int,
    numerical: int,
    communication: int,
    creativity: int,
    leadership: int,
    technical: int,
    helping: int,
    business: int,
    artistic: int,
    stress: int
):

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

    return {"Predicted Career": str(prediction[0])}