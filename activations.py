import numpy as np

def relu(x):
    return max(0,x)
def linear(x):
    return x
def sigmoid(x):
    return 1/(1+np.exp(-x))