# TODO: Import pandas for handling CSV
import pandas as pd

# TODO: Import CountVectorizer and MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Load dataset
# TODO: Load your CSV file "student_notifications.csv"
df = pd.read_csv("student_notifications.csv")

# Step 2: Convert text into numeric features
# TODO: Initialize CountVectorizer
vectorizer = CountVectorizer()

# TODO: Transform the messages into features
X = vectorizer.fit_transform(df["Message"])

# Step 3: Extract target labels
# TODO: Select the target column
y = df["Spam"]

# TODO: Print feature names
print("Feature Names:", vectorizer.get_feature_names_out())

# Step 4: Train the Naive Bayes model
# TODO: Initialize MultinomialNB
model = MultinomialNB()

# TODO: Train the model
model.fit(X, y)

# Step 5: Make predictions
# TODO: Predict on the same dataset
predictions = model.predict(X)
print("Predictions:", predictions.tolist())

# Step 6: Get probability estimates
# TODO: Predict probabilities for each message
probabilities = model.predict_proba(X)

# Step 7: Print probabilities with comments
for msg, row in zip(df["Message"], probabilities):
    not_spam, spam = row
    print(f"Message: {msg}\n[{not_spam:.2f} {spam:.2f}]   # Probabilities\n")
