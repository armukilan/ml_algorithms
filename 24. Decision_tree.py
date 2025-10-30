import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Small dataset
data = {
    "Study_Hours": [2, 3, 4, 5, 6],
    "Attendance": [60, 70, 80, 85, 90],
    "Pass": [0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Separate features and label
X = df[["Study_Hours", "Attendance"]]  # Features
y = df["Pass"]                         # Label



# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)


# Predict on training data
predictions = model.predict(X)
print("Predictions:", predictions.tolist())