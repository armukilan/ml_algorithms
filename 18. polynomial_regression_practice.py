# Polynomial Regression Practice Problem with Plot and Main Function - Fill in the blanks

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def polynomial_regression_task(csv_file="students.csv", degree=2, predict_hours=5, save_plot="output.png"):
    """
    Performs Polynomial Regression on a dataset, predicts score, calculates MSE, and plots the regression curve.
    
    Returns:
        predicted_score (float): Predicted score for given hours
        mse (float): Mean Squared Error of the model
    """
    
    # Step 1: Load the dataset
    df = pd.read_csv("students3.csv")  # Fill CSV filename

    # Step 2: Define features (X) and target (y)
    X = df[["Hours_Studied"]]  # Fill feature column
    y = df["Exam_Score"]    # Fill target column

    # Step 3: Create polynomial features
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    # Step 4: Create the linear regression model
    model = LinearRegression()

    # Step 5: Fit the model
    model.fit(X_poly, y)  # Fill method to train the model

    # Step 6: Make prediction
    predicted_score = model.predict(poly.transform([[predict_hours]]))[0]
    print(f"Predicted exam score for studying {predict_hours} hours is: {predicted_score}")

    # Step 7: Calculate Mean Squared Error
    y_pred = model.predict(X_poly)  # Fill method to get predictions for MSE
    mse = mean_squared_error(y, y_pred)
    print(f"Mean Squared Error of the model: {mse}")

    # Step 8: Plot the data points and polynomial regression curve
    plt.scatter(X, y, color="blue", label="Actual Data")
    
    # Smooth curve for plotting
    X_grid = np.linspace(min(X.values), max(X.values), 100).reshape(-1, 1)
    plt.plot(X_grid, model.predict(poly.transform(X_grid)), color="red", label="Polynomial Regression Curve")
    
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.title("Polynomial Regression: Exam Score vs Hours Studied")
    plt.legend()
    plt.savefig(save_plot)
    plt.close()

    return predicted_score, mse

# Main function to run when script is executed directly
if __name__ == "__main__":
    polynomial_regression_task()
