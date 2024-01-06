# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    """
    load_mnist() function returns a MNIST data
    1st parameter: normalize: normalize inputed image as value from 0.0 to 1.0
        if False, the value is from 0 to 255
    2nd parameter: flatten: flatten inputed image as 1d array
        if False, inputed image saved as 1x28x28 3d array
        if True, inputed image saved as id array with 784 elements
    3rd parameter: one_hot_label: 
        if True, return label as one-hot array
            one-hot array: array e.g. [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                in which the value of correct label is 1 and the others are 0
        if False, return label as integer
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

    """
    normalize setted as True, so the value of each pixel is divided by 255, and the value is between 0.0 and 1.0

    this process of converting data into a specific range is called normalization

    for input date of neural network, this is called "pre-processing"
    pre-processing is helpful for improving the accuracy of neural network and learning speed
    """
    return x_test, t_test


def init_network():
    # in sample_weight.pkl, weight and bias are saved as dictionary
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
        
    """
    output data is an array of probability for each label
    """
    y = softmax(a3)
    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0

for i in range(len(x)):
    y = predict(network, x[i])
    # get the index of the highest probability
    p = np.argmax(y) # 最も確率の高い要素のインデックスを取得

    # compare the index with the correct label
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

"""
the code returns "Accuracy:0.9352"

"""