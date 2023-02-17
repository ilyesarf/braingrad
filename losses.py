import numpy as np
#import activations as acts

def cross_entropy(yHat, y):
    '''
    Calculates the cross entropy between two prbability distributions

    Formula: CE(yHat, y) = -1/N * Σ[i=1; N] y,i * log(yHat,i)

    Args:
        yHat (ndarray (n,)) : estimations
        y (ndarray (n,)) : target values
    '''
    loss = -np.sum(y*np.log(yHat))
    return loss / float(yHat.shape[0]) #normalize loss over samples 

def mean_squared_error(yHat, y):
    '''
    Calculates the mean squared error

    Formula: MSE(yHat, y) = 1/N * Σ[i=1; N] (y,i - yHat,i)^2 

    Args:
        yHat (ndarray (n,)) : estimations
        y (ndarray (n,)) : target values
    '''
    return np.square(np.subtract(y, yHat)).mean()

def mean_absolute_error(yHat, y):
    '''
    Calculates the mean absolute error
    
    Formula: MAE(yHat, y) = 1/N * Σ[i=1; N] |y,i - yHat,i|
    
    Args:
        yHat (ndarray (n,)) : estimations
        y (ndarray (n,)) : target values
    '''
    return np.abs(np.subtract(y, yHat)).mean()
