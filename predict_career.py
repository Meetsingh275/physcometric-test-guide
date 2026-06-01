import pandas as pd
import pickle

print("gihub repo update test")

# Load saved model
with open("career_model.pkl", "rb") as file:
    model = pickle.load(file)

print("\nENTER SCORES BETWEEN 0 TO 100\n")

logical = int(input("Logical Score: "))
numerical = int(input("Numerical Score: "))
communication = int(input("Communication Score: "))
creativity = int(input("Creativity Score: "))
leadership = int(input("Leadership Score: "))
technical = int(input("Technical Interest: "))
helping = int(input("Helping Nature: "))
business = int(input("Business Interest: "))
artistic = int(input("Artistic Interest: "))
stress = int(input("Stress Handling: "))

# Create dataframe
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

# Prediction
prediction = model.predict(sample)

print("\nPredicted Career Option:")
print(prediction[0])