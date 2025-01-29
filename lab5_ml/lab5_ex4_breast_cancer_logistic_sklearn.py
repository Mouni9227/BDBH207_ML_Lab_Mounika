"""Implement logistic regression using scikit-learn for the breast cancer dataset
- https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer  # For imputing missing values
from sklearn.preprocessing import StandardScaler  # For scaling features
import numpy as np

def read_data(file_path):
    # Read dataset
    df = pd.read_csv(file_path)
    print(df.head())
    print(df.shape)
    print(df.info())
    print(df.describe())
    return df

def ploting(data, target):
    # Drop target column for correlation analysis
    data_numeric = data.drop(columns=[target])

    # Plot analysis
    data_numeric.hist(figsize=(10, 8), bins=50, edgecolor='black')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(data_numeric.corr(), annot=True, cmap='YlGn', fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.show()

    sns.pairplot(data, diag_kind='kde', corner=True)
    plt.show()

def preprocess_data(df, target):
    # Drop the 'Unnamed: 32' column (empty or unwanted columns)
    df = df.drop(columns=['Unnamed: 32'], errors='ignore')

    # Encode the target labels: 'M' -> 1, 'B' -> 0
    df[target] = df[target].map({'M': 1, 'B': 0})

    # Separate features and target
    X = df.drop(columns=[target, 'id'])  # Drop the 'id' column if it exists
    y = df[target]

    # Handle missing data
    imputer = SimpleImputer(strategy='mean')  # Use mean imputation
    X_imputed = imputer.fit_transform(X)  # Impute missing values in X

    # Scale the data for better convergence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y

def plot_sigmoid_curve(model, X_test, y_test):
    # Predict probabilities for test data
    y_prob = model.predict_proba(X_test)[:, 1]

    # Plot Sigmoid curve with data points
    plt.figure(figsize=(8, 6))
    sorted_indices = np.argsort(y_prob)  # Sort the probabilities for plotting
    plt.plot(np.sort(y_prob), label="Sigmoid curve", color='blue')
    plt.scatter(range(len(y_prob)), y_prob, color='red', alpha=0.5, label='Data points')
    plt.title("Sigmoid Curve (Predicted Probabilities)")
    plt.xlabel("Samples (sorted by predicted probability)")
    plt.ylabel("Predicted probability")
    plt.legend()
    plt.show()

def main():
    # File path to the dataset
    file_path = "data.csv"  # Adjust this path if necessary
    df = read_data(file_path)

    target = "diagnosis"
    ploting(df, target)

    # Preprocess the data
    X, y = preprocess_data(df, target)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot sigmoid curve
    plot_sigmoid_curve(model, X_test, y_test)

if __name__ == "__main__":
    main()



