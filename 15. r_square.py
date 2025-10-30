import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 1: Create a dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5],
    "Exam_Score": [40, 50, 65, 70, 85]
}
df = pd.DataFrame(data)

# Features (X) and Labels (y)
X = df[["Study_Hours"]]   # Features must be 2D
y = df["Exam_Score"]

# Step 2: Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Step 3: Make predictions
y_pred = model.predict(X)

# Step 4: Calculate R² score
r2 = r2_score(y, y_pred)

print("Actual Scores:", list(y))
print("Predicted Scores:", list(y_pred.round(2)))
print("R² Score:", round(r2, 2))