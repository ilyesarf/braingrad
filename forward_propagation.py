import numpy as np
import activations

# dense = fully connected
def dense_player(input,W,b,activation):
    """
    Computes dense layer
    Args:
      input (ndarray (n, ))
      W    (ndarray (n,j)) : Weight matrix n features per neuron, j neurons
      b    (ndarray (j, )) : bias vector, j neurons  
      g    activation function (sigmoid, relu..)
    Returns
      out (ndarray (j,))  : j units, output for the next layer
    """
    num_neurons=W.shape[1]
    out=np.zeros(num_neurons) # init output as null
    for j in range(num_neurons):
        w=W[:,j] # each column is a specific neuron weights
        z=np.dot(w, input)+b[j]
        out[j]=activation(z)
    return out


# calculates output for each dense layer, for now using only 3 layers
def sequential_model(x, W1, b1, W2, b2, W3, b3): 
    a_out1=dense_player(x,W1,b1,activations.linear)
    a_out2=dense_player(a_out1,W2,b2,activations.linear)
    a_out3=dense_player(a_out2,W3,b3,activations.sigmoid)
    return a_out3


# random data
x=np.random.rand(3) # sample with two features
W1=np.random.rand(3,6) # layer with 6 neurons
b1=np.random.rand(6)
W2=np.random.rand(6,4) # layer with 4 neurons
b2=np.random.rand(4)
W3=np.random.rand(4,1) # layer with single neuron
b3=np.random.rand(1)

print(sequential_model(x, W1, b1, W2, b2, W3, b3))