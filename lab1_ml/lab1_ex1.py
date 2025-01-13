from random import randint
from xml.etree.ElementPath import prepare_descendant
import numpy as np
import matplotlib.pyplot as plt

#Ex1
def main():
    Array=np.array([[1,2,3],[4,5,6]])
    print(Array)
    transpose=np.transpose(Array)
    print(transpose)
    result=np.dot(Array,transpose)
    print(result)
if __name__ == '__main__':

        main()
#Ex2
def line_eqn(x):
    y=(2*x)+3
    return y
x=[x for x in range(-100,100)]
y=[line_eqn(x) for x in range(-100,100)]
line_plot=plt.plot(x,y)

plt.show()

#Ex3
def quadratic_eqn(x):
    y=(2*x)**2 +3*x + 4
    return y
x=[x for x in range(-100,100)]
y=[quadratic_eqn(x) for x in range(-100,100)]
quadratic_plot=plt.plot(x,y)
plt.show()

#Ex4
def guassian_eqn(x,m,s):
    f=(1/(s*np.sqrt(2*np.pi)))*np.exp((-(x-m)**2)/(2*(s)**2))
    return f
x=[x for x in range(-100,100)]
m=np.mean(x)
s=15
y=[guassian_eqn(x=val,m=m,s=s) for val in range(-100,100)]
print(len(x))
print(len(y))
guassian_plot=plt.plot(x,y)
plt.show()

#Ex5
def original_fun(x):
    y1=x**2
    return y1
def grad_fun(x):
   y=np.diff(x)**2
   return y
x=np.arange(-100,100)
y1=original_fun(x)
y2=grad_fun(x)
x_plot=x[:-1]

plt.plot(x,y1)
plt.plot(x_plot,y2)
plt.show()








# import pandas as pd
#
# def calculate_error():
#     def hypothesis(thetas, features, bias):
#         return thetas[0] * bias + sum(t * x for t, x in zip(thetas[1:], features))
#
#     def get_input(prompt, dtype=float, validate=lambda x: True):
#         while True:
#             try:
#                 value = dtype(input(prompt))
#                 if validate(value):
#                     return value
#                 print("Invalid input! Try again.")
#             except ValueError:
#                 print("Invalid input! Try again.")
#
#     # Input for samples, features, bias term, and parameters
#     num_samples = get_input("Enter number of samples: ", int, lambda x: x > 0)
#     num_features = get_input("Enter number of features: ", int, lambda x: x > 0)
#     bias = get_input("Enter the value of bias (x0): ")
#     params = [get_input(f"Enter theta{i}: ") for i in range(num_features + 1)]
#
#     # Collect data
#     data = [[get_input(f"Enter x{i + 1}: ") for i in range(num_features)] + [get_input("Enter Y: ")] for _ in range(num_samples)]
#
#     # Compute error
#     df = pd.DataFrame(data, columns=[f"x{i + 1}" for i in range(num_features)] + ["Y"])
#     total_error = 0.5 * sum((hypothesis(params, row[:-1], bias) - row[-1]) ** 2 for row in df.values)
#
#     print("\nData Table:\n", df)
#     print("\nTotal Squared Error:", total_error)
#
# # Run the function
# calculate_error()











