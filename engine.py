import numpy as np
import math

class Tensor():
	def __init__(self,data,_children=(), _op=''): 
		if type(data).__module__ != np.__name__: # check if data is a numpy array
			data=np.array(data)

		self.data=data
		self.shape = self.data.shape
		self.grad=0.0
		self._backward=lambda: None

		# Keep count of children, aka past tensors we did operations on to get our
		# current tensor to help us implement backward prop
		self._prev=set(_children)   
		self._op=_op

	def __repr__(self): 
		return f"Tensor({self.data}, {self.grad})"

	def __add__(self,other):
		"""
		adds two tensors
		Args:
			self (Tensor)
			other (Tensor)
		Returns
			out (Tensor)  : out.data=self.data+other.data
		"""
		other = other if isinstance(other, Tensor) else Tensor(other)
		out = Tensor(self.data+other.data, (self,other), '+')
		def _backward():
			self.grad+= 1.0*out.grad # chain rule
			other.grad+= 1.0*out.grad
		out._backward=_backward

		return out

	def __mul__(self,other):
		"""
		multiplies two tensors
		Args:
			self (Tensor)
			other (Tensor)
		Returns
			out (Tensor)  : out.data=self.data*other.data
		"""
		other = other if isinstance(other, Tensor) else Tensor(other)
		out = Tensor(self.data*other.data, (self,other), '*')
		def _backward():
			self.grad+=other.data*out.grad
			other.grad+=self.data*out.grad
		out._backward=_backward

		return out

	def tanh(self):
		x=self.data
		t=(np.exp(2*x)-1)/(np.exp(2*x)+1)
		out=Tensor(t,(self,), 'tanh')
		def _backward():
			self.grad+=(1-t**2)*out.grad  # tanh' = 1/cosh^2= 1-tanh^2
		out._backward=_backward

		return out
	
	def backward(self):

		# topological order all of the children in the graph
		topo = []
		visited = set()
		def build_topo(v):
			if v not in visited:
				visited.add(v)
				for child in v._prev:
					build_topo(child)
				topo.append(v)
		build_topo(self)

		# go one variable at a time and apply the chain rule to get its gradient
		self.grad = 1
		for v in reversed(topo):
			v._backward()

if __name__ == "__main__":
    x=Tensor(1)
    y=Tensor(0.6)
    d=(y*x).tanh()

    d.backward()
    print(d)
    print(x)
    print(y)

# L=d*f => dL/dd= f  ;  X=y+b => dX/dy=1.0
# chain rule: dz/dx=dz/dy*dy/dx

#TODO: refactor this