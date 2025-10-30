# Linear Regression Practice Problem with Plot and Main Function - Fill in the blanks

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def linear_regression_task(csv_file="students.csv", predict_hours=5, save_plot="img.png"):
    """
    Performs Linear Regression on a dataset, predicts score, calculates MSE, and plots the regression line.
    Returns:
        predicted_score (float): Predicted score for given hours
        mse (float): Mean Squared Error of the model
    """
    
    # Step 1: Load the dataset
    df = pd.read_csv("students2.csv")  # Fill CSV filename

    # Step 2: Define features (X) and target (y)
    X = df[["Hours_Studied"]]  # Fill feature column
    y = df["Exam_Score"]    # Fill target column

    # Step 3: Create the linear regression model
    model = LinearRegression()

    # Step 4: Fit the model
    model.fit(X, y)  # Fill method to train the model

    # Step 5: Make prediction
    predicted_score = model.predict([[predict_hours]])[0]  # Fill method to predict
    print(f"Predicted exam score for studying {predict_hours} hours is: {predicted_score}")

    # Step 6: Calculate Mean Squared Error
    y_pred = model.predict(X)  # Fill method to get predictions for MSE
    mse = mean_squared_error(y, y_pred)
    print(f"Mean Squared Error of the model: {mse}")

    # Step 7: Plot the data points and regression line
    plt.scatter(X, y, color="blue", label="Actual Data")
    plt.plot(X, y, color="red", label="Regression Line")  # Fill predicted line
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.title("Linear Regression: Exam Score vs Hours Studied")
    plt.legend()
    plt.savefig(save_plot)  # Save the plot as an image
    plt.close()

    return predicted_score, mse

# Main function to run when script is executed directly
if __name__ == "__main__":
    linear_regression_task()
