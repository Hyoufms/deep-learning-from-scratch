"""
perceptron:
    input -> weight -> sum -> activation function -> output
    y = h(b + w1x1 + w2x2)

here h(x) is activation function

activation function:
    input -> activation function (h(x)) -> output

input: x= b + w1x1 + w2x2

activation function: h(x)
    
step function:
h(x) = 0 (x <= 0)
     = 1 (x > 0)

sigmoid function: 
h(x) = 1/(1+exp(-x))

sigmoid function is important for neural network, since it is smooth and differentiable.

both step function and sigmoid function are non-linear functions (非線形関数).

activation function must be non-linear, otherwise neural network cannot be deep.

linear function cannot express complex data.

ReLU (Rectified Linear Unit):  recenly most popular activation function
h(x) = x (x > 0)  # when input is positive, output is same as input
     = 0 (x <= 0) # when input is negative, output is 0

"""

import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

def step_function_2(x):
    y = x > 0
    return y.astype(np.integer) # bool -> int

def show_step_function_graphic():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function_2(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1) # define y axis range
    plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def show_sigmoid_function_graphic():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1) # define y axis range
    plt.show()

def ReLU(x):
    return np.maximum(0, x)

def show_ReLU_function_graphic():
    x = np.arange(-5.0, 5.0, 0.1)
    y = ReLU(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 5.1) # define y axis range
    plt.show()

if __name__ == "__main__":
    show_step_function_graphic()
    show_sigmoid_function_graphic()
    show_ReLU_function_graphic()

