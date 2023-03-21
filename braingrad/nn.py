import random
from engine import Tensor
class Unit():  # unit = neuron
    def __init__(self, nin):
        self.w = Tensor([random.uniform(-1,1) for dummy in range(nin)])  # weights
        self.b = Tensor(random.uniform(-1,1))  # bias
    def __call__(self, x):
        # W@x + b,  @ is dot product
        return self.w.dot(x) + self.b
class Layer():
    def __init__(self, nin, nout, act):
        self.units = [Unit(nin) for dummy in range(nout)]
        self.act = act
    def __call__(self, x):
        return Tensor([self.act(unit(x)).data for unit in self.units])
class MLP():
    def __init__(self, nin, nouts, acts):
        nouts=[nin]+nouts
        self.layers = [Layer(nouts[i], nouts[i+1], acts[i]) for i in range(len(nouts)-1)]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
x=[2.0,3.6]
u1=Unit(2)
l=Layer(2,3,lambda x: x)
m=MLP(2,[3,2],[lambda x: x, lambda x: x])
print(m(x))

