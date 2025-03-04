import pandas as pd

def ordinal_encoding(df, columns):
    mappings = {}
    for column in columns:
        df[column] = df[column].astype(str)  # Ensure categorical values are strings
        unique_vals = sorted(df[column].unique())
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        df[column] = df[column].map(mapping)
        mappings[column] = mapping
    return df, mappings

def main():
    file_path = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
    df = pd.read_csv(file_path, header=None)
    df.columns = ["age", "menopause", "tumor_size", "inv_nodes", "node_caps",
                  "deg_malig", "breast", "breast_quad", "irradiat", "class"]

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("Before Ordinal Encoding:")
    print(df.head(10))

    # Define categorical columns for ordinal encoding (excluding `deg_malig`)
    categorical_cols = ["age", "menopause", "tumor_size", "inv_nodes", "node_caps","deg_malig","breast", "breast_quad", "irradiat", "class"]

    df_encoded, mappings = ordinal_encoding(df, categorical_cols)

    print("\nAfter Ordinal Encoding:")
    print(df_encoded.head(10))

    print("\nMappings Used for Encoding:")
    for col, mapping in mappings.items():
        print(f"{col}: {mapping}")

if __name__ == "__main__":
    main()
