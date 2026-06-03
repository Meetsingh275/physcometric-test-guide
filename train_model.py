import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pickle

data = pd.read_csv("psychometric_results.csv")

X = data.drop(["StudentName", "CareerOption"], axis=1)
y = data["CareerOption"]

model = RandomForestClassifier(n_estimators=100, random_state=42)

scores = cross_val_score(model, X, y, cv=5)
print(f"Average Accuracy: {scores.mean()*100:.2f}%")

# Final training on full dataset
model.fit(X, y)

with open("career_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model Trained and Saved Successfully")
