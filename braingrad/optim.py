
class SGD:
    def __init__(self, tensors, lr=0.01):
        self.tensors = tensors
        self.lr = lr

    def step(self):
        for tensor in self.tensors:
            #print(f"{'#'*8} layer {self.tensors.index(tensor)+1} grad {'#'*8}\n{self.lr * tensor.grad}\n")
            tensor.data -= self.lr * tensor.grad