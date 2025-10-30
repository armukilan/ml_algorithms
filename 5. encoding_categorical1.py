import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Load dataset
df = pd.read_csv("employees2.csv")
print("Original Data:")
print(df)

# Step 2: Label Encoding (Gender) using LabelEncoder
label_encoder = LabelEncoder()
df["Gender_Label"] = label_encoder.fit_transform(df["Gender"])
print("\nAfter Label Encoding Gender:")
print(df[["Name", "Gender", "Gender_Label"]])

# Step 3: One-Hot Encoding (Department)
df_onehot = pd.get_dummies(df, columns=["Department"])
print("\nAfter One-Hot Encoding Department:")
print(df_onehot)
