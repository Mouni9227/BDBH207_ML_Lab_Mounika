"""Use validation set to do feature and model selection."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

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
def gradient_descent(X, y, theta, alpha=0.01, threshold=1e-2, max_iterations=10000):
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
    return (X - X.mean(axis=0)) / X.std(axis=0)

# Function to generate polynomial features
def generate_polynomial_features(X, degree):
    poly = PolynomialFeatures(degree)
    return poly.fit_transform(X)

# Load dataset
file_path = "simulated_data_multiple_linear_regression_for_ML.csv"
data = pd.read_csv(file_path)
target_col = "disease_score"

# Feature-target separation
X = data.drop(columns=target_col).values
y = data[target_col].values

# Standardize features
X = standardize(X)

# Add bias term
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Define the number of folds (K)
k = 10

# Get the fold indices
fold_indices = kfold_indices(X, k)

# Initialize list to store results
r2_scores = []

# Iterate through each fold
for train_indices, test_indices in fold_indices:
    X_train_full, y_train_full = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # Further split training data into training and validation sets (80-20 split)
    split_index = int(len(X_train_full) * 0.8)
    X_train, X_val = X_train_full[:split_index], X_train_full[split_index:]
    y_train, y_val = y_train_full[:split_index], y_train_full[split_index:]

    best_r2 = -np.inf
    best_theta = None
    best_degree = 1

    # Try polynomial degrees from 1 to 4
    for degree in range(1, 5):
        X_train_poly = generate_polynomial_features(X_train, degree)
        X_val_poly = generate_polynomial_features(X_val, degree)
        X_test_poly = generate_polynomial_features(X_test, degree)

        # Initialize model parameters
        theta = np.zeros(X_train_poly.shape[1])

        # Train the model using gradient descent
        theta, _ = gradient_descent(X_train_poly, y_train, theta, alpha=0.01)

        # Validate the model
        y_val_pred = hypothesis(X_val_poly, theta)
        val_r2 = r2_score_manual(y_val, y_val_pred)

        # Keep track of the best model
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_theta = theta
            best_degree = degree

    # Evaluate the best model on the test set
    X_test_poly = generate_polynomial_features(X_test, best_degree)
    y_test_pred = hypothesis(X_test_poly, best_theta)
    test_r2 = r2_score_manual(y_test, y_test_pred)
    r2_scores.append(test_r2)

# Calculate the mean R² score across all folds
mean_r2_score = np.mean(r2_scores)
print("K-Fold Cross-Validation R² Scores:", r2_scores)
print("Mean R² Score:", mean_r2_score)

# --- Scikit-Learn Implementation for Comparison ---
kf = KFold(n_splits=10, shuffle=True, random_state=42)
sklearn_r2_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    sklearn_r2_scores.append(r2_score(y_test, y_pred))

print("Scikit-Learn K-Fold Cross-Validation R² Scores:", sklearn_r2_scores)
print("Scikit-Learn Mean R² Score:", np.mean(sklearn_r2_scores))
