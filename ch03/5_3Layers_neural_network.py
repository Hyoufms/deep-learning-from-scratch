import numpy as np

"""
this is a 3 layers neural network
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

def init_network():
    """
    initialize wieghts and biases
    """
    network = {}
    # layer 1
    network['w1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b2'] = np.array([0.1, 0.2, 0.3])

    # layer 2
    network['w2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])

    # layer 3
    network['w3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    """
    processing from input toward output
    """

    w1, w2, w3 = network['w1'], network['w2'], network['w3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(x, w2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot()   
    y = identity_function(a3)

    return y

def main():
    network = init_network()
    x = np.array([1.0, 0.5]) # input
    y = forward(network, x) # output
    print(y)

if __name__ == '__main__':
    main()