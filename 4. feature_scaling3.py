import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
# TODO: Read the CSV file "employees.csv"
df = pd.read_csv("employees1.csv")
# exit()
print(df.head())

# Step 2: Select numeric feature(s) to scale (Salary and Experience)
# TODO: Select Salary and Experience columns
X = ['Salary', 'Experience']
# exit()

# Step 3: Apply Standard Scaling
# TODO: Create a StandardScaler object
scaler = StandardScaler()

# TODO: Fit and transform the data using scaler
X_scaled = scaler.fit_transform(df[X])

# Step 4: Convert back to DataFrame
# TODO: Convert the scaled NumPy array back to a DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=["Salary", "Experience"])

# Step 5: Print the scaled values
print("Standardized Data:")
print(X_scaled_df)
