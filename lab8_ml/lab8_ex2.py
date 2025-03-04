"""Build a classification model for wisconsin dataset using Ridge and Lasso classifier using scikit-learn"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def read_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(axis=1, how='all')  # Drop columns with all NaN values
    return df

def plot_results(model, X_test_scaled, y_test, title):
    y_prob = model.decision_function(X_test_scaled)
    plt.figure(figsize=(8, 6))
    sorted_indices = np.argsort(y_prob)
    plt.plot(np.sort(y_prob), label="Decision Boundary", color='blue')
    plt.scatter(range(len(y_prob)), y_prob, color='red', alpha=0.5, label='Data Points')
    plt.title(title)
    plt.xlabel("Samples (sorted by decision function value)")
    plt.ylabel("Decision Function Value")
    plt.legend()
    plt.show()


def main():
    file_path = '/home/ibab/Downloads/wisconsin.csv'
    data = read_data(file_path)

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    drop_col = ["diagnosis", "id"]
    X = data.drop(columns=drop_col, errors='ignore')
    y = data["diagnosis"]

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge_clf = RidgeClassifier(alpha=10.0)
    ridge_clf.fit(X_train_scaled, y_train)
    ridge_predictions = ridge_clf.predict(X_test_scaled)
    ridge_accuracy = accuracy_score(y_test, ridge_predictions)
    print(f'Ridge Classifier Accuracy: {ridge_accuracy:.2f}')
    plot_results(ridge_clf, X_test_scaled, y_test, "Ridge Classifier Decision Boundary")

    lasso_clf = LogisticRegression(penalty='l1', solver='saga', max_iter=10000)
    lasso_clf.fit(X_train_scaled, y_train)
    lasso_predictions = lasso_clf.predict(X_test_scaled)
    lasso_accuracy = accuracy_score(y_test, lasso_predictions)
    print(f'Lasso Classifier Accuracy: {lasso_accuracy:.2f}')
    plot_results(lasso_clf, X_test_scaled, y_test, "Lasso Classifier Decision Boundary")


if __name__ == '__main__':
    main()

