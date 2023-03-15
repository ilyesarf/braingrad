
class Optimizer:
    def __init__(self, tensors):
        self.tensors = tensors
    
    def zero_grad(self):
        for tensor in self.tensors:
            tensor.grad = None

class SGD(Optimizer):
    def __init__(self, tensors, lr=0.01):
        super().__init__(tensors)
        self.tensors = tensors
        self.lr = lr

    def step(self):
        for tensor in self.tensors:
            tensor.data -= self.lr * tensor.grad