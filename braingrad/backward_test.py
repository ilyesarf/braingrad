import sys
import numpy as np
sys.path.insert(1, '..')
from braingrad.engine import Tensor
from braingrad.optim import SGD
import torch

l1_init = np.random.randn(784, 128).astype(np.float32) 
l2_init = np.random.randn(128, 10).astype(np.float32)
x_init = np.random.randn(128, 784).astype(np.float32)
y_init = np.random.randn(128, 10).astype(np.float32)
def test_braingrad():
  x = Tensor(x_init)
  l1 = Tensor(l1_init)
  l2 = Tensor(l2_init)
  y = Tensor(y_init)
  optim = SGD([l1, l2])
  out = x.dot(l1)
  outr = out.relu()
  out_l2 = outr.dot(l2)
  outl = out_l2.logsoftmax()
  outm = outl.mul(y)
  outx = outm.mean()
  outx.backward()
  optim.step()

test_braingrad()
"""
def test_pytorch():
  x = torch.tensor(x_init, requires_grad=True)
  W = torch.tensor(W_init, requires_grad=True)
  m = torch.tensor(m_init)
  out = x.matmul(W)
  outr = out.relu()
  outl = torch.nn.functional.log_softmax(outr, dim=1)
  outm = outl.mul(m)
  outa = outm.add(m)
  outx = outa.sum()
  outx.backward()
  return outx.detach().numpy(), x.grad, W.grad

for x,y in zip(test_braingrad(), test_pytorch()):
    print(x)
    print(y)
    print()
    np.testing.assert_allclose(x, y, atol=1e-6)
"""
