import numpy as np

def neural_network_matrix_multiplaction():
    X = np.array([1, 2])
    W = np.array([[1, 3, 5], [2 , 4, 6]])
    print(X, 'shape', X.shape) # [1 2] shape (2,)
    print(W, 'shape', W.shape) # [[1 3 5] [2 4 6]] shape (2, 3)
    Y = np.dot(X, W)
    print(Y) # [ 5 11 17]

if __name__ == '__main__':
    neural_network_matrix_multiplaction()