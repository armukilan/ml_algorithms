import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Step 1: Prepare the data
experience = [[1], [2], [3], [5], [7], [10]]   # Years of experience
salary = [30, 35, 50, 80, 95, 105]            # Salary (in $1000s)

# Step 2: Transform input into polynomial features (degree=2 → quadratic curve)
poly = PolynomialFeatures(degree=2)
experience_poly = poly.fit_transform(experience)

# Step 3: Create and train the model
model = LinearRegression()
model.fit(experience_poly, salary)

# Step 4: Make a prediction (e.g., 6 years of experience)
predicted_salary = model.predict(poly.transform([[6]]))
print(predicted_salary)   # Output: [~85]