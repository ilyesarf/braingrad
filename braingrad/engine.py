import numpy as np

class Tensor():
	def __init__(self,data,_children=(), _op='', requires_grad=True): 
		if type(data).__module__ != np.__name__: # check if data is a numpy array
			if isinstance(data, (int, float)):
				data = [data]
			data=np.array(data, dtype=np.float32)
		
		self.data=data
		self.shape = self.data.shape
		self.requires_grad = requires_grad #not working
		self.grad=None
		self._backward=lambda: None

		# Keep count of children, aka past tensors we did operations on to get our
		# current tensor to help us implement backward prop
		self._prev=set(_children)   
		self._op=_op

	def __repr__(self): 
		return f"Tensor({self.data}, grad={self.grad})"

	def astype(self, type):
		self.data = self.data.astype(type)

	############ generators ############

	_rng = np.random.default_rng()
	
	@staticmethod
	def random(shape, **kwargs):
		return Tensor(Tensor._rng.random(shape, **kwargs))
	
	@staticmethod
	def uniform(low, high, **kwargs):
		return Tensor(Tensor._rng.uniform(low, high, **kwargs))
	
	@staticmethod
	def eye(shape, **kwargs):
		return Tensor(np.eye(shape, **kwargs))

	############ main ops ############
	
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
			self.grad = 1.0*out.grad # chain rule
			other.grad = 1.0*out.grad
		out._backward=_backward

		return out
	
	def dot(self, other):
		other = other if isinstance(other, Tensor) else Tensor(other)	
		
		out = Tensor(self.data.dot(other.data), (self,other), 'dot')

		def _backward():
			self.grad = out.grad.dot(other.data.T) if self.grad is None else (self.grad + out.grad.dot(other.data.T))
			other.grad = out.grad.T.dot(self.data).T if other.grad is None else (other.grad + out.grad.T.dot(self.data).T)
			
		out._backward = _backward

		return out
	
	def mul(self, other):
		other = other if isinstance(other, Tensor) else Tensor(other)	
		out = Tensor(self.data*other.data, (self,other), '*')
		def _backward():
			self.grad=out.grad*other.data
			other.grad=out.grad*self.data
		out._backward = _backward

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
		
		if any(len(i) < 2 for i in (self.shape, other.shape)): #checks if there's a scalar
			out = self.mul(other)
		else:
			out = self.dot(other)
			
		return out
	
	def __pow__(self, other):
		"""
		raises tensor to an int/float power (tensor coming soon)
		Args:
			self (Tensor)
			other (int or float)
		Returns:
			out (Tensor)  : out.data=self.data**other
		"""

		#other = other if isinstance(other, Tensor) else Tensor(other)
		assert isinstance(other, (int, float)), "power should be int or float"
		out = Tensor(self.data**other, (self,), f'**{other}')		

		def _backward():
			self.grad =  (other * self.data**(other-1)) * out.grad
		out._backward = _backward

		return out

	############ sub-ops ############
	
	def __neg__(self): # -self
		return self * -1

	def __radd__(self, other): # other + self
		return self + other

	def __sub__(self, other): # self - other
		return self + (-other)

	def __rsub__(self, other): # other - self
		return other + (-self)

	def __rmul__(self, other): # other * self
		return self * other

	def __truediv__(self, other): # self / other
		return self * other**-1

	def __rtruediv__(self, other): # other / self
		return other * self**-1
	
	def square(self):
		return self.__pow__(2)
	
	############ complementary ops ############

	def log(self):
		out = Tensor(np.log(self.data), (self,), 'log')
		def _backward():
			self.grad = 1/self.data
		out._backward = _backward

		return out

	def mean(self):
		out = Tensor(self.data.mean(), (self,), 'mean')
		def _backward():
			self.grad = np.full_like(self.data, 1/self.data.size)
		out._backward = _backward

		return out
	
	def sum(self):
		out = Tensor(np.sum(self.data), (self,), 'sum')
		def _backward():
			self.grad = np.ones_like(self.data)
		out._backward = _backward

		return out
	
	def abs(self):
		out = Tensor(np.abs(self.data), (self,), 'abs')
		def _backward():
			self.grad = np.sign(out.data)
		out._backward = _backward

		return out
	
	############ activations ############
	
	def relu(self):
		out = Tensor(np.maximum(0, self.data), (self,), 'relu')
		def _backward():
			grad = out.grad.copy()
			grad[self.data < 0] = 0
			self.grad = grad
		out._backward = _backward

		return out

	def sigmoid(self):
		def _sigmoid(x):
			return 1/(1+np.exp(-x))
		out = Tensor(_sigmoid(self.data), (self,), 'sigmoid')
		
		def _backward():
			self.grad = out.grad * (1-out.data)
		out._backward = _backward

		return out
	
	def logsoftmax(self):
		def logsumexp(x):
			c = x.max(axis=1)
			logsumexp = np.log(np.exp(x - c.reshape((-1, 1))).sum(axis=1))
			return (c + logsumexp).reshape(-1, 1)
		
		out = Tensor(self.data - logsumexp(self.data), (self,), 'softmax')

		def _backward():
			self.grad = out.grad - np.exp(out.data)*out.grad.sum(axis=1).reshape((-1, 1))
		out._backward = _backward

		return out
	
	def tanh(self):
		def _tanh(x):
			return np.exp(2*x)-1/(np.exp(2*x)+1)
		out=Tensor(_tanh(self.data),(self,), 'tanh')

		def _backward():
			self.grad=(1-out.data**2)*out.grad  # tanh' = 1/cosh^2= 1-tanh^2
		out._backward=_backward

		return out
	
	#backward
	def backward(self):
		assert len(self.shape) == 0, "grad can only be created for scalar outputs"
		# topological order all of the children in the graph
		topo = []
		visited = set()
	
		self.grad = np.ones_like(self.data)
		def build_topo(v):
			if v not in visited:
				for child in v._prev:
					build_topo(child)
				topo.append(v)
		build_topo(self)
		# go one variable at a time and apply the chain rule to get its gradient
		for v in reversed(topo):
			v._backward()


