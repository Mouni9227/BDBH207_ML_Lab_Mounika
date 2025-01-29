"""Implement normal equations method from scratch and compare your results on a simulated dataset (disease score fluctuation as target)
and the admissions dataset (https://www.kaggle.com/code/erkanhatipoglu/linear-regression-using-the-normal-equation ).
You can compare the results with scikit-learn and your own gradient descent implementation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to read and display data
def read_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    print(f"Data Shape: {df.shape}")
    print("\nData Info:")
    print(df.info())
    print("\nFirst Five Rows:")
    print(df.head())
    print("\nData Description:")
    print(df.describe())
    return df

# Perform Exploratory Data Analysis (EDA)
def eda_analysis(df, target_cols):
    print("Performing EDA...")
    # Histograms for numeric columns
    df.hist(figsize=(12, 8), bins=30, edgecolor="black")
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    # Pairplot for relationships
    sns.pairplot(df, diag_kind="kde", corner=True)
    plt.show()

# Train-test split function
def train_test_split(X, y, test_size=0.3):
    split_index = int(len(X) * (1 - test_size))
    X_train = X[:split_index].reset_index(drop=True)
    X_test = X[split_index:].reset_index(drop=True)
    y_train = y[:split_index]
    y_test = y[split_index:]
    return X_train, X_test, y_train, y_test

# Calculate R² score manually
def r2_score_manual(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Plot actual vs predicted values
def plot_actual_vs_predicted(y_actual, y_predicted):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_actual, y=y_predicted, alpha=0.7)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], "r--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.show()

# Normal equation function
def normal_equation(X, y):
    X_transpose = np.transpose(X)
    theta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y
    return theta

# Main function
def main():
    # Load data
    file_path = "simulated_data_multiple_linear_regression_for_ML.csv"
    data = read_data(file_path)

    # EDA
    target_cols = ["disease_score", "disease_score_fluct"]
    eda_analysis(data, target_cols)

    # Feature-target separation
    X = data.drop(columns=target_cols)
    y = data["disease_score"].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Standardize features
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()

    # Add bias term to features
    X_train.insert(0, "Bias", 1)
    X_test.insert(0, "Bias", 1)

    # Train model using the normal equation
    theta = normal_equation(X_train.values, y_train)

    # Validate model
    y_pred_test = X_test.values @ theta
    r2_score = r2_score_manual(y_test, y_pred_test)
    print(f"R² Score: {r2_score:.4f}")

    # Plot actual vs predicted
    plot_actual_vs_predicted(y_test, y_pred_test)

    print(f"Updated theta values: {theta}")

if __name__ == "__main__":
    main()

"""
Normal Equation Results:
R² Score: 0.9843
Updated theta values: [8.15176119e+02 4.28728297e+01 4.12219779e+00 1.56330860e+01
 1.59192849e+01 5.05505235e-01]

Gradient descent Results:
Gradient descent converged after 898 iterations.
R² Score: 0.5478
Updated theta values: [813.84968114  41.22279589   7.34429223  19.18087938  13.75742183
 -10.23409396]

Interpretation from Both the methods of obtaining parameters for target variable(Disease score fluct) :
1.Given dataset is Small in size: (60,7)
2.r2 score for Normal equation: 0.9843
3.r2 score for Gradient decent: 0.5478
4.The normal equation outperforms gradient descent due to 
the small dataset size and its ability to compute the exact solution. 
5.Avoided iterative optimization, leading to precise and direct results in normal equation method.
6.Gradient decent Not ideal for large datasets due to the high computational cost of O(n3)O(n3) for matrix inversion.
7.The lower R2R2 score from gradient descent suggests that the algorithm didn’t fully converge to the optimal solution or encountered difficulties.
"""
