"""
machine learning solves classification problem and regression problem
classification problem: classify data into several classes
regression problem: predict value from input data

"""

import numpy as np

"""
@param: a is an array

softmax function
which outputs number between 0 and 1
and sum of all output is 1
therefore, its output can be interpreted as probability
"""
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) # prevent overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


if __name__ == '__main__':
    pass
