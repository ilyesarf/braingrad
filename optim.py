
class SGD:
    def __init__(self, tensors, lr=0.01):
        self.tensors = tensors
        self.lr = lr
    
    def step(self):
        for tensor in self.tensors:
            tensor.data -= self.lr * self.grad