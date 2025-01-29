"""Implement Stochastic Gradient Descent algorithm from scratch"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def h_fun(X, thetas):
    return np.dot(X, thetas.T)
def cost_fun(X, y, thetas):
    y_pred = h_fun(X, thetas)
    cost = (1 / len(y)) * np.sum((y_pred - y) ** 2)
    return cost
def mormalization(X):
    X_norm = (X - X.min()) / (X.max() - X.min())
    return X_norm
def SGD(X, y, thetas, n_iters, alpha):
    cost_history = [0] * n_iters
    for i in range(n_iters):
        for j in range(len(y)):
            rand_index = np.random.randint(len(y))
            ind_x = X[rand_index:rand_index + 1]
            ind_y = y[rand_index:rand_index + 1]
            gradients = 2 * ind_x.T.dot(h_fun(ind_x, thetas) - ind_y)
            eta = alpha  # Use the constant learning rate
            thetas = thetas - eta * gradients
        cost_history[i] = cost_fun(X, y, thetas)  # Record cost for the current iteration
    return thetas, cost_history

def main():
    file_path = "simulated_data_multiple_linear_regression_for_ML.csv"  #
    data = read_data(file_path)
    target_cols = ["disease_score", "disease_score_fluct"]
    X = data.drop(target_cols, axis=1)
    y = data["disease_score_fluct"]
    thetas = np.zeros(X.shape[1])

    initial_cost = cost_fun(X, y, thetas)
    print("Initial cost:", initial_cost)

    X = mormalization(X)
    alpha = 0.01

    th_n, cost_history = SGD(X, y, thetas, 250, alpha)

    final_cost = cost_fun(X, y, th_n)
    print("Mean squared error after training:", final_cost)
    print("\nUpdated thetas after training:", th_n)
    y_pred = h_fun(X, th_n)

    # Displaying actual vs predicted scores for the first few entries
    print("\nActual vs Predicted scores (first 10 samples):")
    for i in range(10):
        print(f"Actual: {y.iloc[i]}, Predicted: {y_pred[i]}")

    # Plot the cost history
    plt.plot(range(250), cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('SGD-Cost Function vs Iterations')
    plt.show()

if __name__ == "__main__":
    main()