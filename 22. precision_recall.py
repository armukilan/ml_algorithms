from sklearn.metrics import precision_score, recall_score

# Actual labels (ground truth)
y_true = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]

# Predictions from our model
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print("Precision:", precision)
print("Recall:", recall)