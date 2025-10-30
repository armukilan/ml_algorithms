import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("employees3.csv")   # Fill in the CSV filename

def split_employee_data():
    # Step 1: Load dataset

    # Step 2: Features (Experience, Salary) and Label (Promoted)
    X = df[["Experience", "Salary"]]           # Fill in the feature columns
    y = df["Promoted"]             # Fill in the label column

    # Step 3: Split into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # Fill in X and y
    )

    # Step 4: Return the results(training and testing sets)
    return X_train, X_test, y_train, y_test


# Example usage
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_employee_data()
    print("Training set:")
    print(X_train)
    print(y_train)
    print("\nTesting set:")
    print(X_test)
    print(y_test)
