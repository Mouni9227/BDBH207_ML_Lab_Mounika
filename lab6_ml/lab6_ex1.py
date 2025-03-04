"""K-fold cross validation. Implement for K = 10. Implement from scratch, then, use scikit-learn methods."""

import numpy as np
import pandas as pd

# Function to generate K-fold indices
def kfold_indices(data, k):
    fold_size = len(data) // k
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    return folds

# Function to compute hypothesis
def hypothesis(X, theta):
    return np.dot(X, theta)

# Function to compute the cost
def compute_cost(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    errors = predictions - y
    return (1 / (2 * m)) * np.sum(errors ** 2)

# Function for gradient descent
def gradient_descent(X, y, theta, alpha, threshold=1e-2, max_iterations=10000):
    m = len(y)
    cost_history = []
    iteration = 0

    while iteration < max_iterations:
        gradient = (1 / m) * np.dot(X.T, (hypothesis(X, theta) - y))
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        if iteration > 0 and abs(cost_history[-2] - cost) < threshold:
            break

        iteration += 1

    return theta, cost_history

# Function to calculate R² score manually
def r2_score_manual(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Function to standardize features
def standardize(X):
    return (X - X.mean()) / X.std()

# Load data
file_path = "simulated_data_multiple_linear_regression_for_ML.csv"
data = pd.read_csv(file_path)
target_col = "disease_score"

# Feature-target separation
X = data.drop(columns=target_col)
y = data[target_col].values

# Add bias term to features
X.insert(0, "Bias", 1)
X = X.values

# Define the number of folds (K)
k = 10

# Get the fold indices
fold_indices = kfold_indices(X, k)

# Initialize a list to store the R² scores
r2_scores = []

# Iterate through each fold
for train_indices, test_indices in fold_indices:
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # Standardize features
    X_train = standardize(X_train)
    X_test = standardize(X_test)

    # Initialize model parameters
    theta = np.zeros(X_train.shape[1])

    # Train the model using gradient descent
    alpha = 0.01  # Learning rate
    theta, cost_history = gradient_descent(X_train, y_train, theta, alpha)

    # Make predictions on the test data
    y_pred = hypothesis(X_test, theta)

    # Calculate the R² score for this fold
    r2_score = r2_score_manual(y_test, y_pred)
    r2_scores.append(r2_score)

# Calculate the mean R² score across all folds
mean_r2_score = np.mean(r2_scores)
print("K-Fold Cross-Validation R² Scores:", r2_scores)
print("Mean R² Score:", mean_r2_score)




