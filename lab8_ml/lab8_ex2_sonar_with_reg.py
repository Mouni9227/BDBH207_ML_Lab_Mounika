import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def read_data(file_path):
    df = pd.read_csv(file_path, header=None)  # Sonar dataset has no headers
    return df

def plot_coefficients(coefs, title):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(coefs)), coefs, color='blue', alpha=0.7)
    plt.xlabel("Feature Index")
    plt.ylabel("Coefficient Value")
    plt.title(title)
    plt.show()


def main():
    file_path = "/home/ibab/Downloads/sonar data.csv"  # Update path if needed
    data = read_data(file_path)

    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]  # Labels (last column)

    # Convert labels to binary (Sonar dataset uses 'R' and 'M' for Rock and Mine)
    y = y.map({'R': 1, 'M': 0})

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_clf = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=10000)
    log_clf.fit(X_train_scaled, y_train)
    log_predictions = log_clf.predict(X_test_scaled)
    log_accuracy = accuracy_score(y_test, log_predictions)

    print(f'Logistic Regression Accuracy (Ridge Regularization): {log_accuracy:.2f}')
    print(f"Logistic Regression Coefficients:\n{log_clf.coef_[0]}")

    plot_coefficients(log_clf.coef_[0], "Logistic Regression Coefficients (Ridge Regularization)")


if __name__ == '__main__':
    main()
