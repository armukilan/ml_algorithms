from sklearn.linear_model import LinearRegression

# Step 1: Prepare the data
hours = [[1], [2], [3], [4]]       # input feature: Study Hours
scores = [40, 50, 65, 70]         # output/label: Exam Scores

# Step 2: Create the model
model = LinearRegression()

# Step 3: Train the model (find the best line)
model.fit(hours, scores)

# Step 4: Make a prediction
predicted_score = model.predict([[3.5]])
print(predicted_score)  # Output: [66] (approx)