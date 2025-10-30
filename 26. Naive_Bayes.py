import pandas as pd

# Step 1: Create the Dataset
data = {
    "Message": [
        "Win a free lottery ticket",
        "Free money offer just for you",
        "Meeting at 10am tomorrow",
        "Project deadline extended",
        "Congratulations you won a prize"
    ],
    "Spam": [1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

# Step 2: Convert Text into Features
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Message"])  # Features
y = df["Spam"]                               # Labels

print("Feature Names:", vectorizer.get_feature_names_out(), "\n")

# Step 3: Train a Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X, y)

# Step 4: Make Predictions
predictions = model.predict(X)
print("Predictions:", predictions.tolist(), "\n")

# Step 5: Probability of Predictions
probabilities = model.predict_proba(X)
for msg, row in zip(df["Message"], probabilities):
    not_spam, spam = row
    print(f"Message: {msg}")
    print(f"[{not_spam:.2f} {spam:.2f}]   # {int(not_spam*100)}% Not Spam, {int(spam*100)}% Spam\n")
