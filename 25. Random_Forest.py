import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


data = {
    "Study_Hours": [2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 8, 9],
    "Attendance": [60, 65, 70, 75, 80, 62, 68, 72, 78, 85, 88, 90, 95],
    "Pass":        [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["Study_Hours", "Attendance"]]
y = df["Pass"]

model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=3, 
    min_samples_leaf=2, 
    random_state=42
)
model.fit(X, y)
predictions = model.predict(X)

# Predict probabilities
probabilities = model.predict_proba(X)

# Round to 1 decimal
probabilities = np.round(probabilities, 1)

#Print the Predicted Probabilities
print("Predictions:", predictions.tolist())

# Print with comments
for row in probabilities:
    fail, pass_ = row
    print(f"[{fail:.1f} {pass_:.1f}]   # {int(fail*100)}% Fail, {int(pass_*100)}% Pass")
