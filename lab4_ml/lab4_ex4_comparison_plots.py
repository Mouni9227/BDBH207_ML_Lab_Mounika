# Plot the data points and the obtained regression line from all three approaches and compare the outcome.
# Function to plot regression lines and compare results
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from lab3_ml.lab3_ex2_simulated_gradientd_scratch import hypothesis


def plot_regression_lines(X, y, theta_gd, theta_ne, model_sklearn, X_feature_name):
    plt.figure(figsize=(10, 6))

    # Actual data points
    plt.scatter(X[:, 1], y, color="blue", alpha=0.6, label="Data Points")

    # Gradient descent regression line
    y_pred_gd = hypothesis(X, theta_gd)
    plt.plot(X[:, 1], y_pred_gd, color="orange", label="Gradient Descent Line")

    # Normal equation regression line
    y_pred_ne = hypothesis(X, theta_ne)
    plt.plot(X[:, 1], y_pred_ne, color="green", label="Normal Equation Line")

    # Scikit-learn regression line
    y_pred_sklearn = model_sklearn.predict(X[:, 1].reshape(-1, 1))
    plt.plot(X[:, 1], y_pred_sklearn, color="red", label="Scikit-Learn Line")

    plt.xlabel(X_feature_name)
    plt.ylabel("Disease Score Fluctuation")
    plt.title("Regression Lines: Gradient Descent, Normal Equation, and Scikit-Learn")
    plt.legend()
    plt.show()


def main():
    # Load data
    file_path = "simulated_data_multiple_linear_regression_for_ML.csv"
    data = pd.read_csv(file_path)

    # Separate features and target
    X = data[["age"]].values  # Use the age feature for visualization
    y = data["disease_score_fluct"].values

    # Add bias term for gradient descent and normal equation
    X_b = np.c_[np.ones(X.shape[0]), X]

    # Gradient descent theta (intercept and age coefficient only)
    theta_gd = np.array([813.84968114,0])  # Intercept + coefficient for age

    # Normal equation theta (intercept and age coefficient only)
    theta_ne = np.array([815.176119,0])  # Intercept + coefficient for age

    # Scikit-learn model
    model_sklearn = LinearRegression()
    model_sklearn.fit(X, y)

    # Plot regression lines
    plot_regression_lines(X_b, y, theta_gd, theta_ne, model_sklearn, "Age")

if __name__ == "__main__":
    main()
