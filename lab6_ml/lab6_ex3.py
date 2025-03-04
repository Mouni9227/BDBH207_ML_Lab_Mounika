"""Data standardization - scale the values such that mean of new dist = 0 and sd = 1. Implement code from scratch."""

import numpy as np
import pandas as pd

# Function to standardize data
def standardize_data(data):
    mean_vals = data.mean(axis=0)
    std_vals = data.std(axis=0)
    standardized_data = (data - mean_vals) / std_vals
    return standardized_data

def main():
    file_path = "simulated_data_multiple_linear_regression_for_ML.csv"
    data = pd.read_csv(file_path)
    standardized_data = standardize_data(data)
    print(standardized_data.head())


if __name__ == '__main__':
    main()
