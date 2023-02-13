import numpy as np
#import activations as acts

def cross_entropy(yHat, y):
    '''
    Calculates the cross entropy between two prbability distributions

    Formula: CE(yHat, y) = -1/N * sum(y * log(yHat))

    Args:
        yHat (ndarray (n,)) : estimations
        y (ndarray (n,)) : target values
    '''
    loss = -np.sum(y*np.log(yHat))
    return loss / float(yHat.shape[0]) #normalize loss over samples 
