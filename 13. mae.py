from sklearn.metrics import mean_absolute_error

# Actual vs Predicted values
y_true = [50, 60, 70, 80]
y_pred = [48, 65, 68, 75]

mae = mean_absolute_error(y_true, y_pred)
print("MAE:", mae)
