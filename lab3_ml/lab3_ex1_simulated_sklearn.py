'''Implement a linear regression model using scikit-learn for
the simulated dataset - simulated_data_multiple_linear_regression_for_ML.csv
 - to predict the disease score from multiple clinical parameters '''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

# Display dataset information and the first few rows
print("Dataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Exploratory Data Analysis (EDA)
print("\nSummary Statistics:")
pd.set_option('display.max_columns', None)
print(df.describe())
pd.reset_option('display.max_columns')

# Plot histograms for all numerical columns
df.hist(figsize=(12, 10), bins=30, edgecolor="black")
plt.subplots_adjust(hspace=0.7, wspace=0.4)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Scatterplot matrix for the first few features
sns.pairplot(df, diag_kind="kde", corner=True)
plt.show()

# --------------------- Box Plots for each variable and target ---------------------
# Box plot for each feature and target variable
plt.figure(figsize=(15, 12))
features = df.drop(["disease_score", "disease_score_fluct"], axis=1)  # Excluding target variables from the features

# Plot boxplots for all features
for i, column in enumerate(features.columns, 1):
    plt.subplot(3, 4, i)  # Assuming there are 12 features for 3 rows and 4 columns grid
    sns.boxplot(x=df[column])
    plt.title(f"Box plot for {column}")

# Box plot for target variables 'disease_score' and 'disease_score_fluct'
plt.subplot(3, 4, len(features.columns) + 1)
sns.boxplot(x=df["disease_score"])
plt.title("Box plot for disease_score")

plt.subplot(3, 4, len(features.columns) + 2)
sns.boxplot(x=df["disease_score_fluct"])
plt.title("Box plot for disease_score_fluct")

plt.tight_layout()
plt.show()

# --------------------- Case 1: Predicting 'disease_score' ---------------------
# Exclude 'disease_score_fluct' as a feature
X = df.drop(["disease_score", "disease_score_fluct"], axis=1)  # Features
y = df["disease_score"]  # Target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Linear Regression for 'disease_score' (excluding 'disease_score_fluct'):")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-Squared: {r2:.2f}")

# Plot actual vs. predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Disease Score")
plt.ylabel("Predicted Disease Score")
plt.title("Actual vs Predicted Disease Scores (Excluding 'disease_score_fluct')")
plt.show()

# --------------------- Case 2: Predicting 'disease_score_fluct' ---------------------
# Exclude 'disease_score' as a feature
X = df.drop(["disease_score", "disease_score_fluct"], axis=1)  # Features
y = df["disease_score_fluct"]  # Target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a linear regression model
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nLinear Regression for 'disease_score_fluct' (excluding 'disease_score'):")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-Squared: {r2:.2f}")

# Plot actual vs. predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Disease Score Fluctuation")
plt.ylabel("Predicted Disease Score Fluctuation")
plt.title("Actual vs Predicted Disease Score Fluctuation (Excluding 'disease_score')")
plt.show()