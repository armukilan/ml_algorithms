import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the dataset
df = pd.read_csv("students.csv")

# Step 2: Select numeric feature(s) to scale (Age and Marks)
data = ['Age', 'Marks']

# Step 3: Apply Min-Max Scaling
scalar = MinMaxScaler()
X_scaled = scalar.fit_transform(df[data])  # Fit and transform the selected columns


# Step 4: Convert back to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=data)

# Step 5: Print the scaled values
print("Scaled Data:")
print(X_scaled_df)
