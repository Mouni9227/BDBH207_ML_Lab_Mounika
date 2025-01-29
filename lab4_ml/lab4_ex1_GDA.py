#GRADIENT DESCENT ALGORITHM IMPLEMENTATION
import numpy as np

# Hypothesis function
def hypothesis(X, theta):
    return np.dot(X, theta)

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    errors = predictions - y
    return (1 / (2 * m)) * np.sum(errors ** 2)

# Gradient Descent function
def gradient_descent(X, y, theta, alpha, threshold=1e-4, max_iterations=10000):
    m = len(y)  # Number of training examples
    cost_history = []  # To track the cost at each iteration
    iteration = 0  # Iteration counter

    while iteration < max_iterations:
        # Compute the gradient (partial derivatives of the cost function w.r.t theta)
        gradient = (1 / m) * np.dot(X.T, (hypothesis(X, theta) - y))

        # Update the parameters (theta)
        theta -= alpha * gradient

        # Calculate the cost after the update
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        # Stop if the change in cost is below the threshold
        if iteration > 0 and abs(cost_history[-2] - cost) < threshold:
            break

        iteration += 1

    print(f"Gradient descent converged after {iteration} iterations.")
    return theta, cost_history



