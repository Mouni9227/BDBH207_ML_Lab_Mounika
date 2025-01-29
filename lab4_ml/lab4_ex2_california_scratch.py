"""Use your implementation and train ML models for both california housing and simulated datasets and
compare your results with the scikit-learn models. """


"""Need improvement for below code r2 score is 0.54, feature scaling , regularization , one-hot encoding are required """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to read and display data
def read_data(file_path):
    # Read data from the CSV file
    df = pd.read_csv(file_path)
    # Set pandas options for displaying output
    pd.set_option('display.max_rows', 6)
    pd.set_option('display.max_columns', 11)
    # Display first few rows of the data
    print("First 5 rows of the data:")
    print(df.head())
    # Show data shape and info
    print("Shape:", df.shape)
    print("Info:", df.info())
    # # Handle missing values in 'total_bedrooms' column: fill with median
    # if 'total_bedrooms' in df.columns:
    #     df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    #
    # # Handle categorical 'ocean_proximity' column: One-Hot Encoding
    # if 'ocean_proximity' in df.columns:
    #     df = pd.get_dummies(df, columns=['ocean_proximity'],
    #                         drop_first=True)  # One-hot encode and drop the first category

    # Reset pandas options for display
    pd.set_option('display.max_rows', None)

    # Display summary statistics
    print("Data Description:")
    print(df.describe())

    return df
def eda_analysis(df, target_cols):
    print("Performing EDA...")
    # Histograms for numeric columns
    df.hist(figsize=(12, 8), bins=30, edgecolor="black")
    plt.tight_layout()
    plt.show()
    # Correlation heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    # plt.title("Correlation Heatmap")
    # plt.show()
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
# Hypothesis function
def hypothesis(X, theta):
    return np.dot(X, theta)
# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    errors = predictions - y
    return (1 / (2 * m)) * np.sum(errors ** 2)
# Gradient descent function
def gradient_descent(X, y, theta, alpha, threshold=1e-4, max_iterations=40000):
    m = len(y)
    cost_history = []
    iteration = 0

    while iteration < max_iterations:
        gradient = (1 / m) * np.dot(X.T, (hypothesis(X, theta) - y))
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        # Stop if cost difference between iterations is less than the threshold
        if iteration > 0 and abs(cost_history[-2] - cost) < threshold:
            break

        iteration += 1

    print(f"Gradient descent converged after {iteration} iterations.")
    return theta, cost_history
# Standardize features
def standardize(X):
    # # Check if the features have NaN values before standardizing
    # if X.isnull().sum().any():
    #     print("Warning: NaN values in data. Filling NaNs with column mean.")
    #     X = X.fillna(X.mean())
    return (X - X.mean()) / X.std()

# Calculate R² score manually
def r2_score_manual(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Plot the cost function over iterations
def plot_cost(cost_history):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(cost_history)), cost_history, color="blue")
    plt.title("Cost Function Convergence")
    plt.xlabel("Iterations")
    plt.ylabel("Cost (J)")
    plt.show()

# Plot actual vs predicted values and print a sample of actual vs predicted values
def plot_actual_vs_predicted(y_actual, y_predicted):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_actual, y=y_predicted, alpha=0.7)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], "r--", label="Ideal Line")
    plt.xlabel("Actual Values for house price")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.show()

    # Print actual vs predicted values
    comparison_df = pd.DataFrame({'Actual': y_actual, 'Predicted': y_predicted})
    print("\nSample of Actual vs Predicted values:")
    print(comparison_df.head())  # Show first few rows of comparison

# Main function
def main():
    file_path = "housing.csv"
    data = read_data(file_path)

    # Define target columns and drop unnecessary columns
    target_cols = ["median_house_value"]
    drop_cols = ["median_house_value","total_bedrooms", "ocean_proximity"]

    # Feature and target separation
    X = data.drop(columns=drop_cols)
    y = data["median_house_value"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Standardize features
    X_train = standardize(X_train)
    X_test = standardize(X_test)

    # Add bias term to features
    X_train.insert(0, "Bias", 1)
    X_test.insert(0, "Bias", 1)

    # Initialize model parameters
    theta = np.zeros(X_train.shape[1])
    alpha = 0.01  # Learning rate

    # Train model using gradient descent
    theta, cost_history = gradient_descent(X_train.values, y_train, theta, alpha)

    # Plot cost history
    plot_cost(cost_history)

    # Validate model
    y_pred_test = hypothesis(X_test.values, theta)
    r2_score_value = r2_score_manual(y_test, y_pred_test)
    print(f"R² Score: {r2_score_value:.4f}")

    # Plot actual vs predicted
    plot_actual_vs_predicted(y_test, y_pred_test)

    eda_analysis(X_train, target_cols)

    # Print updated theta values
    print(f"Updated theta values: {theta}")


if __name__ == "__main__":
    main()

