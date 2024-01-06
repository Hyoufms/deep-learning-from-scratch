import numpy as np

# def AND(x1, x2):
#     w1, w2, thetha = 0.5, 0.5, 0.7
#     temp = x1*w1 + x2*w2
#     if temp <= thetha:
#         return 0
#     elif temp > thetha:
#         return 1
    
def AND(x1, x2 , bias):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = bias
    temp = np.sum(w*x) + b
    if temp <= 0:
        return 0
    else:
        return 1
    
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    temp = np.sum(w*x) + b
    if temp <= 0:
        return 0
    else:
        return 1
    
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    temp = np.sum(w*x) + b
    print(temp)
    if temp <= 0:
        return 0
    else:
        return 1
    
def XOR(x1, x2): # XOR gate can be made by combining AND, NAND, OR gates
    s1 = NAND(x1,x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

def main():
    # print(AND(0, 0)) # 0을 출력
    # print(AND(1, 0)) # 0을 출력
    # print(AND(0, 1)) # 0을 출력
    # print(AND(1, 1)) # 1을 출력

    print("AND")
    print(AND(1, 1, -0.7))
    print("NAND")
    print(NAND(1, 1))
    print("OR")
    print(OR(1, 0))

if __name__ == "__main__":
    main()

"""
perceptrons are algorithm with inputs and outputs
bias and wieghts are parameters of perceptrons

AND, NAND, OR gates can be made by single perceptron
XOR gate can be made by combining AND, NAND, OR gates

"""