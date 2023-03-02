import numpy as np
from engine import Tensor

def relu(x):
    return max(0,x)
def linear(x):
    return x
def sigmoid(x):
    return 1/(1+np.exp(-x))
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def tanh(tensor):
	x=tensor.data
	t=(np.exp(2*x)-1)/(np.exp(2*x)+1)
	out=Tensor(t,(tensor,), 'tanh')
	def _backward():
		tensor.grad+=(1-t**2)*out.grad  # tanh' = 1/cosh^2= 1-tanh^2
	out._backward=_backward

	return out