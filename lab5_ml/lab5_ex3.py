"""Compute the derivative of a sigmoid function and visualize it"""
import numpy as np
from matplotlib import pyplot as plt

def sigmoid_fun(data):
    return 1/(1+np.exp(-data))

def der_sigmoid(data):
    return data * (1 -data)

def main():
    data = np.linspace(-10,5, 100)
    sig_val = sigmoid_fun(data)
    der_sig_val = der_sigmoid(sig_val)

    plt.subplot(1, 1, 1)
    plt.plot(data, sig_val, label="s-curve")
    plt.plot(data, der_sig_val, label="d-curve")
    plt.show()

if __name__ == '__main__':
    main()
