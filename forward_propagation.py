import numpy as np
import activations

# dense = fully connected
def dense_layer(input,W,b,activation):
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


# calculates output for each dense layer
def sequential_model(x, all_W, all_B, n_layers): 
    """
    Args:
        x (ndarray (n, ))
        all_W (array of ndarrays (n, )) : A list of all weights
        all_B (array of ndarrays (n, )) : A list of all biases
        n_layers (int) : Number of layers in NN
    """

    for i in range(n_layers):
        if i != n_layers-1:
            act = activations.linear
        else:
            act = activations.sigmoid

        x = dense_layer(x,all_W[i],all_B[i],act)
    
    return x 

if __name__ == '__main__':
    # random data

    n_layers = 3 # 3 layers for testing

    x=np.random.rand(3) # sample with two features
    W1=np.random.rand(3,6) # layer with 6 neurons
    b1=np.random.rand(6)
    W2=np.random.rand(6,4) # layer with 4 neurons
    b2=np.random.rand(4)
    W3=np.random.rand(4,1) # layer with single neuron
    b3=np.random.rand(1)

    print(sequential_model(x, [W1, W2, W3], [b1, b2, b3], n_layers))
