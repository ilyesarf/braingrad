import sys
sys.path.insert(1, '..')
import numpy as np
from braingrad.nn import Unit, Layer, MLP
from braingrad.engine import Tensor

def test_unit():
    u1=Unit(2)
    assert u1.w.data.shape == (2,)
    assert u1.b.data.shape == (1,)
    assert u1([2.0,3.6]).data.shape == (1,)
    assert u1([2.0,3.6]).data == u1.w.dot([2.0,3.6]).data + u1.b.data
def test_layer():
    l=Layer(2,3,lambda x: x)
    assert l([2.0,3.6]).data.shape == (3,)
    assert l([2.0,3.6]).data[0] == l.units[0]([2.0,3.6]).data
    assert l([2.0,3.6]).data[1] == l.units[1]([2.0,3.6]).data
    assert l([2.0,3.6]).data[2] == l.units[2]([2.0,3.6]).data
def test_mlp():
    m=MLP(2,[3,2],[lambda x: x, lambda x: x])
    assert m([2.0,3.6]).data.shape == (2,)
    assert m([2.0,3.6]).data[0] == m.layers[1](m.layers[0]([2.0,3.6])).data[0]
    assert m([2.0,3.6]).data[1] == m.layers[1](m.layers[0]([2.0,3.6])).data[1]