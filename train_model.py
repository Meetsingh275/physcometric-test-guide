import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
data = pd.read_csv("psychometric_results.csv")
X = data.drop(["StudentName", "CareerOption"], axis=1)
y = data["CareerOption"]
model = DecisionTreeClassifier()
model.fit(X, y)
with open("career_model.pkl", "wb") as file:
    pickle.dump(model, file)
    print("Model Trained Successfully")  