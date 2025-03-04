import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def read_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(axis=1, how='all')
    return df

def plot_coefficients(coefs, title):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(coefs)), coefs, color='blue', alpha=0.7)
    plt.xlabel("Feature Index")
    plt.ylabel("Coefficient Value")
    plt.title(title)
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

    # Ridge Regularization (L2)
    ridge_clf = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000)
    ridge_clf.fit(X_train_scaled, y_train)
    ridge_predictions = ridge_clf.predict(X_test_scaled)
    ridge_accuracy = accuracy_score(y_test, ridge_predictions)
    print(f'Logistic Regression Accuracy (Ridge - L2 Regularization): {ridge_accuracy:.2f}')
    print(f"Ridge Coefficients:\n{ridge_clf.coef_[0]}")
    plot_coefficients(ridge_clf.coef_[0], "Logistic Regression Coefficients (Ridge - L2)")

    # Lasso Regularization (L1)
    lasso_clf = LogisticRegression(penalty='l1', solver='saga', max_iter=10000)
    lasso_clf.fit(X_train_scaled, y_train)
    lasso_predictions = lasso_clf.predict(X_test_scaled)
    lasso_accuracy = accuracy_score(y_test, lasso_predictions)
    print(f'Logistic Regression Accuracy (Lasso - L1 Regularization): {lasso_accuracy:.2f}')
    print(f"Lasso Coefficients:\n{lasso_clf.coef_[0]}")
    plot_coefficients(lasso_clf.coef_[0], "Logistic Regression Coefficients (Lasso - L1)")

if __name__ == '__main__':
    main()
