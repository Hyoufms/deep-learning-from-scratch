import numpy as np

"""
1-demensional array
"""

def demo_1d_array():
    A = np.array([1, 2, 3, 4]) 
    print(A) # [1 2 3 4]

    print('dimension', np.ndim(A)) # 1 (dimension)
    print('shape', A.shape) # (4,)   (shape)

def demo_2d_array():
    B = np.array([[1, 2], [3, 4], [5, 6]])
    print(B)  
    """
    [[1 2]
    [3 4]
    [5 6]]
    """
    print('shape', B.shape) # (3, 2) (3 rows, 2 columns)

"""
matrix multiplication
"""
def matrix_multiplication():
    A = np.array([[1, 2], [3, 4]])
    print('shape', A.shape) # (2, 2)
    B = np.array([[5, 6], [7, 8]])
    print('shape', B.shape) # (2, 2)

    print('multiplication', np.dot(A, B))
    """
    [[19 22]
    [43 50]]
    """

def matrix_multiplication_2x3_3x2():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    print('shape', A.shape) # (2, 3)
    B = np.array([[1, 2], [3, 4], [5, 6]])
    print('shape', B.shape) # (3, 2)

    print('multiplication', np.dot(A, B))
    """
    [[22 28]
    [49 64]]
    """
# number of A's rows elements must be same as B's columns elements
    
def matrix_multiplication_3x2_2x1():
    A = np.array([[1, 2], [3, 4], [5, 6]])
    print('shape', A.shape) # (3, 2)
    B = np.array([7, 8])
    print('shape', B.shape) # (2,)

    print('multiplication', np.dot(A, B))
    """
    [23 53 83]
    """

if __name__ == '__main__':
    # demo_1d_array()
    # demo_2d_array()
    # matrix_multiplication()
    matrix_multiplication_2x3_3x2()
    matrix_multiplication_3x2_2x1()