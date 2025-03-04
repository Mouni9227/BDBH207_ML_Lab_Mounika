"""Implement ordinal encoding and one-hot encoding methods in Python from scratch."""
import pandas as pd

def one_hot_encoding(df, nominal_cols):
    one_hot = pd.get_dummies(df[nominal_cols], drop_first=False)
    df = df.drop(columns=nominal_cols)
    df = pd.concat([df, one_hot], axis=1)
    return df

def main():
    file_path = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
    df = pd.read_csv(file_path, header=None)
    df.columns = ["age", "menopause", "tumor_size", "inv_nodes", "node_caps","deg_malig", "breast", "breast_quad", "irradiat", "class"]

    nominal_cols = ["node_caps", "breast", "breast_quad", "irradiat"]

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("Before One-Hot Encoding:")
    print(df.head())
    print(df.shape)

    df = one_hot_encoding(df, nominal_cols)

    print("After One-Hot Encoding (Original Nominal Columns Removed):")
    print(df.head())
    print(df.shape)

if __name__ == "__main__":
    main()

