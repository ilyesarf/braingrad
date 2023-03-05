from engine import Tensor

def cross_entropy(yHat, y):
    if not isinstance(yHat, Tensor):
        yHat = Tensor(yHat)
    
    if not isinstance(y, Tensor):
        y = Tensor(y)
    '''
    Calculates the cross entropy between two prbability distributions

    Formula: CE(yHat, y) = -1/N * Σ[i=1; N] y,i * log(yHat,i)

    Args:
        yHat (Tensor) : estimations
        y (Tensor) : target values
    '''
    loss = (y*yHat.log()).mean()*-1
    return loss

def mean_squared_error(yHat, y):
    if not isinstance(yHat, Tensor):
        yHat = Tensor(yHat)
    
    if not isinstance(y, Tensor):
        y = Tensor(y)
    '''
    Calculates the mean squared error

    Formula: MSE(yHat, y) = 1/N * Σ[i=1; N] (y,i - yHat,i)^2 

    Args:
        yHat (Tensor) : estimations
        y (Tensor) : target values
    '''
    loss = (y-yHat).square().mean()
    return loss

def mean_absolute_error(yHat, y):
    if not isinstance(yHat, Tensor):
        yHat = Tensor(yHat)
    
    if not isinstance(y, Tensor):
        y = Tensor(y)

    '''
    Calculates the mean absolute error
    
    Formula: MAE(yHat, y) = 1/N * Σ[i=1; N] |y,i - yHat,i|
    
    Args:
        yHat (Tensor) : estimations
        y (Tensor) : target values
    '''
    loss = (y-yHat).abs().mean()

    return loss
