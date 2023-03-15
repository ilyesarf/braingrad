#!/usr/bin/env python3
import sys
sys.path.insert(1, '..')
import numpy as np
from fetch import fetch_mnist
from braingrad.engine import Tensor
from braingrad.optim import SGD
from tqdm import trange

# load the mnist dataset

X_train = fetch_mnist("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_train = fetch_mnist("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch_mnist("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch_mnist("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

# train a model

def layer_init(m, h):
  ret = (Tensor.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)).astype(np.float32)
  return ret

class BrainNet:
  def __init__(self):
    self.l1 = layer_init(784, 128)
    self.l2 = layer_init(128, 10)

  def forward(self, x):
    
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()


def train(model, optim, BS):

    for i in trange(1000):
      
      samp = np.random.randint(0, X_train.shape[0], size=(BS))
      x = Tensor(X_train[samp].reshape((-1, 28*28)))
      # forward prop
      outs = model.forward(x)

      # labels
      Y = Y_train[samp]
      y = np.zeros((len(samp),10), np.float32)
      y[range(y.shape[0]),Y] = -1.0
      y = Tensor(y)

      # NLL loss function
      loss = outs.mul(y).mean()

      optim.zero_grad()
      loss.backward()
      optim.step()


if __name__ == '__main__':
    model = BrainNet()
    optim = SGD([model.l1, model.l2])

    BS = 128

    train(model, optim, BS)
    # evaluate

    def numpy_eval():
      Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28))))
      Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
      return (Y_test == Y_test_preds).mean()

    accuracy = numpy_eval()
    print("test set accuracy is %f" % accuracy)

    assert accuracy > 0.95
