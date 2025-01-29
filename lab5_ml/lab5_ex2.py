"""Implement sigmoid function in python and visualize it"""
import numpy as np
from matplotlib import pyplot as plt

def sigmoid_fun(data):
    val = 1/(1+np.exp(-data))
    return val

def main():
    data = np.linspace(-10, 10, 100)
    output = sigmoid_fun(data)

    plt.figure()
    plt.plot(data, output)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == '__main__':
    main()