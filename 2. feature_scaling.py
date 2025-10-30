import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Step 1: Create a simple dataset
data = {
    "Age": [18, 25, 40, 50, 60],
    "Salary": [20000, 50000, 100000, 150000, 200000]
}
df = pd.DataFrame(data)
print("Original Data:\n", df)

# Step 2: Apply Min-Max Scaling (values between 0 and 1)
minmax_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df), columns=df.columns)
print("\nAfter Min-Max Scaling:\n", df_minmax)

# Step 3: Apply Standardization (mean = 0, std = 1)
standard_scaler = StandardScaler()
df_standard = pd.DataFrame(standard_scaler.fit_transform(df), columns=df.columns)
print("\nAfter Standardization:\n", df_standard)
