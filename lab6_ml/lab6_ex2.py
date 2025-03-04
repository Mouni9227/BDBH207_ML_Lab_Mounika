"""Data normalization - scale the values between 0 and 1. Implement code from scratch."""

import numpy as np
import pandas as pd

def normalize_data(data):
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

def main():
    file_path = "simulated_data_multiple_linear_regression_for_ML.csv"
    data = pd.read_csv(file_path)
    normalized_data = normalize_data(data)
    print(normalized_data.head())

if __name__ == '__main__':
    main()
