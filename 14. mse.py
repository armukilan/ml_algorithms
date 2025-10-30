import math
from sklearn.metrics import mean_squared_error

# Step 1: Actual and Predicted Scores
y_true = [50, 60, 70, 80]      # Actual values
y_pred = [48, 65, 68, 75]      # Predicted values

# Step 2: Calculate MSE using scikit-learn
mse = mean_squared_error(y_true, y_pred)

# Step 3: Calculate RMSE (square root of MSE)
rmse = math.sqrt(mse)

print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))