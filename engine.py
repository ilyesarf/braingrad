import numpy as np

class Tensor():
	def __init__(self,data,_children=(), _op=''): 
		if type(data).__module__ != np.__name__: # check if data is a numpy array
			if isinstance(data, (int, float)):
				data = [data]
			data=np.array(data, dtype=np.float32)

		self.data=data
		self.shape = self.data.shape
		self.grad=0.0
		self._backward=lambda: None

		# Keep count of children, aka past tensors we did operations on to get our
		# current tensor to help us implement backward prop
		self._prev=set(_children)   
		self._op=_op

	def __repr__(self): 
		return f"Tensor({self.data}, grad={self.grad})"

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

		if self.shape[0] != other.shape[0] and self.shape[0] == other.data.T.shape[0]:
			x = self.data
			y = other.data.T
		else:
			x = self.data
			y = other.data
		
		if any(len(i) < 2 for i in (x.shape, y.shape)): #checks if there's a scalar
			out = Tensor(x*y, (self,other), '*')
			def _backward():
				self.grad+=out.grad*y
				other.grad+=(out.grad.T*x).T
				if other.grad.shape != other.shape:
					other.grad = other.grad.T
		else:
			out = Tensor(x.dot(y), (self,other), '*')

			def _backward():
				self.grad+=out.grad.dot(y)
				#if self.grad.shape != self.shape:
				#	self.grad = np.reshape(self.grad, self.shape)
				other.grad+=out.grad.T.dot(x).T
				if other.grad.shape != other.shape:
					other.grad = other.grad.T
				
		out._backward=_backward

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
			self.grad +=  (other * self.data**(other-1)) * out.grad
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
	
	############ activations ############
	
	def relu(self):
		out = Tensor(np.maximum(0, self.data), (self,), 'relu')
		def _backward():
			self.grad += 1*(out.data<=0)
		out._backward = _backward

		return out

	def sigmoid(self):
		def _sigmoid(x):
			return 1/(1+np.exp(-x))
		out = Tensor(_sigmoid(self.data), (self,), 'sigmoid')
		
		def _backward():
			self.grad += out.data * (1-out.data)
		out._backward = _backward

		return out
	
	def softmax(self):
		def _softmax(x):
			return np.exp(x)/np.sum(np.exp(x))
		out = Tensor(_softmax(self.data), (self,), 'sigmoid')

		def _backward():
			return

		return out
	
	def tanh(self):
		def _tanh(x):
			return np.exp(2*x)-1/(np.exp(2*x)+1)
		out=Tensor(_tanh(self.data),(self,), 'tanh')

		def _backward():
			self.grad+=(1-out.data**2)*out.grad  # tanh' = 1/cosh^2= 1-tanh^2
		out._backward=_backward

		return out
	
	#backward
	def backward(self):
		assert len(self.shape) == 0, "grad can only be created for scalar outputs"
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
		for v in reversed(topo):
			v._backward()

if __name__ == "__main__":
	x = Tensor(np.eye(3))
	print(f'x data: \n{x.data}\n')
	y = Tensor([[2, 0, -2]])
	print(f'y data: \n{y.data}\n')
	
	m = y*x
	print(f'm data: \n{m.data}]\n')
	z = m.sum()
	print(f'z data: \n{z.data}\n')
	z.backward()

	print(f'x grad: \n{x.grad}\n')  # dz/dx
	print(f'y grad: \n{y.grad}')  # dz/dy

# L=d*f => dL/dd= f  ;  X=y+b => dX/dy=1.0
# chain rule: dz/dx=dz/dy*dy/dx
