import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
    df = pd.read_csv(url, header=None)
    df.columns = ["age", "menopause", "tumor_size", "inv_nodes", "node_caps", "deg_malig", "breast", "breast_quad", "irradiat", "class"]
    return df

def preprocess_data(df):
    df["deg_malig"] = df["deg_malig"].astype(str).str.replace("'", "").astype(int)

    label_encoder = LabelEncoder()
    df["class"] = label_encoder.fit_transform(df["class"])

    ordinal_cols = ["age", "menopause", "tumor_size", "inv_nodes"]
    ordinal_encoder = OrdinalEncoder()
    df[ordinal_cols] = ordinal_encoder.fit_transform(df[ordinal_cols])

    nominal_cols = ["node_caps", "breast", "breast_quad", "irradiat"]
    one_hot_encoder = OneHotEncoder(drop="first", sparse_output=False)
    one_hot_encoded = one_hot_encoder.fit_transform(df[nominal_cols])

    # Convert One-Hot Encoded data to DataFrame
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(nominal_cols))

    # Merge encoded data and drop original columns
    df = df.drop(columns=nominal_cols)
    df = pd.concat([df, one_hot_df], axis=1)

    # Ensuring  all columns are numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Check for any conversion issues
    if df.isna().sum().sum() > 0:
        print("Warning: Some values could not be converted to numbers!")
        print(df.dtypes)

    return df, label_encoder

def train_logistic_regression(df):
    X = df.drop(columns=["class"])
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy:.2f}")
    return model

def main():
    df = load_data()
    print("Before Encoding & Processing:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.head(20))
    df, label_encoder = preprocess_data(df)
    print("After Encoding & Processing:")
    print(df.head(20))
    model = train_logistic_regression(df)

if __name__ == "__main__":
    main()
