from sklearn.metrics import accuracy_score

# Actual labels (ground truth)
y_true = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]

# Predictions from our model
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]


acc = accuracy_score(y_true, y_pred)
print("Accuracy:", acc)