import numpy as np

class Tensor():
  def __init__(self,data,_children=(), _op=''): 

    if type(data).__module__ != np.__name__: # check if data is a numpy array
        data=np.array(data)
    self.data=data

    # Keep count of children, aka past tensors we did operations on to get our
    # current tensor to help us implement backward prop
    self._prev=set(_children)   
    self._op=_op

  def shape(self):
    return self.data.shape

  def __repr__(self): 
    return f"Tensor({self.data})"

  def __add__(self,other):
    return Tensor(self.data+other.data, (self,other), '+')
    
  def __mul__(self,other):
    return Tensor(self.data*other.data, (self,other), '*')

x=Tensor((5,6,32,9,51,0))
print(x)
