import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load training data
data = pd.read_csv("Titanic.csv")

# Keep only weaker features for baseline
features = ["Pclass", "Sex"]
target = "Survived"
data = data[features + [target]]

# Map categorical
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

# Drop missing values if any (Age is removed anyway)
data = data.dropna()

# Split features and target
X = data[features]
y = data[target]

# Train logistic regression with slight regularization to lower accuracy
model = LogisticRegression(max_iter=500, C=0.1, random_state=42)
model.fit(X, y)

# Predict on training set for demonstration
preds = model.predict(X)

# Compute baseline accuracy
accuracy = accuracy_score(y, preds)
print(f"Baseline Logistic Regression Accuracy: {accuracy * 100:.2f}%")  
