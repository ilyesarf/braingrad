#!/usr/bin/env python3
import sys
sys.path.insert(1, '..')
import numpy as np
from braingrad.engine import Tensor
from braingrad.optim import SGD
from tqdm import trange

# load the mnist dataset

def fetch(url):
  import requests, gzip, os, hashlib, numpy
  fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp):
    with open(fp, "rb") as f:
      dat = f.read()
  else:
    with open(fp, "wb") as f:
      dat = requests.get(url).content
      f.write(dat)
  return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()
X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

# train a model

def layer_init(m, h):
  ret = Tensor.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
  ret.astype(np.float32)
  return ret

class BrainNet:
  def __init__(self):
    self.l1 = layer_init(784, 128)
    self.l2 = layer_init(128, 10)

  def forward(self, x):
    
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()


model = BrainNet()
optim = SGD([model.l1, model.l2])

BS = 128

for i in (t := trange(1000)):
  
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

  # is it accurate?
  pred = np.argmax(outs.data, axis=1)
  accuracy = (pred == Y).mean()

# evaluate

def numpy_eval():
  Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28))))
  Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
  return (Y_test == Y_test_preds).mean()

accuracy = numpy_eval()
print("test set accuracy is %f" % accuracy)

assert accuracy > 0.95