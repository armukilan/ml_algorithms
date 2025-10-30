import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
df = pd.read_csv("customers.csv")
print("Original Data:")
print(df)

# Step 2: Label Encoding (Subscribed)
label_encoder = LabelEncoder()
# Convert the "Subscribed" column into numeric values using LabelEncoder
df["Subscribed_Label"] = label_encoder.fit_transform(df["Subscribed"])
print("\nAfter Label Encoding Subscribed:")
print(df[["Name", "Subscribed", "Subscribed_Label"]])

# Step 3: One-Hot Encoding (Region)
# Convert the "Region" column into multiple binary columns
df_onehot = pd.get_dummies(df, columns=["Region"])

# Step 4: Display results
print("\nAfter One-Hot Encoding Region:")
print(df_onehot)
