import sys
sys.path.insert(1, '..')
import numpy as np
import torch
from engine import Tensor

X = Tensor.random((3,3))
print(f'x data: \n{X}\n')
Y = Tensor.random((1,3))
print(f'y data: \n{Y}\n')

def test_backward():
    def braingrad():
        print("\n\n******* BRAINGRAD *******\n\n")
        x = X
        y = Y

        m = y.dot(x)
        
        v = y.logsoftmax().sum()

        v.backward()

        print(f'm data: \n{m.data}]\n')
        z = m.sum()
        print(f'z data: \n{z.data}\n')
        z.backward()

        print(f'x grad: \n{x.grad}\n')  # dz/dx
        print(f'y grad: \n{y.grad}')  # dz/dy
    
        return x.grad, y.grad
    
    def pytorch():
        print("\n\n******* TORCH *******\n\n")
        x = torch.tensor(X.data, requires_grad=True)
        y = torch.tensor(Y.data, requires_grad=True)

        m = y.matmul(x)
        
        v = torch.nn.functional.log_softmax(y, dim=1).sum()

        v.backward()

        print(f'm data: \n{m.data}]\n')
        z = m.sum()
        print(f'z data: \n{z.data}\n')
        z.backward()

        print(f'x grad: \n{x.grad}\n')  # dz/dx
        print(f'y grad: \n{y.grad}')  # dz/dy
    
        return x.grad, y.grad
    
    for a,b in zip(braingrad(), pytorch()):
        np.testing.assert_allclose(a, b, atol=1e-6)