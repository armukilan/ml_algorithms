import pandas as pd

# TODO import the Linear Regression model
from sklearn.linear_model import LinearRegression

# Step 1: Load dataset
df = pd.read_csv("employee_study.csv")   # TODO: Fill in the CSV filename

# Step 2: Select feature and label
X = df[["Hours_Studied"]]              # TODO: Fill in feature column
y = df["Exam_Score"]                # TODO: Fill in label column

# Step 3: Create the model
# TODO: Initialize LinearRegression model
model = LinearRegression()

# Step 4: Train the model
# TODO: Train the model using fit()
model.fit(X,y)

# Step 5: Make prediction
predicted_score = model.predict([[6]])   # Predict score for 6 hours
print(predicted_score)