from braingrad.engine import Tensor

def cross_entropy(yHat, y):
    if not isinstance(yHat, Tensor):
        yHat = Tensor(yHat)
    
    if not isinstance(y, Tensor):
        y = Tensor(y)
    '''
    Calculates the cross entropy between two probability distributions

    Args:
        yHat (Tensor) : estimations
        y (Tensor) : target values
    Returns:
        loss (Tensor) : -1/N * Σ[i=1; N] y,i * log(yHat,i)
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

    Args:
        yHat (Tensor) : estimations
        y (Tensor) : target values
    Returns:
        loss (Tensor) : 1/N * Σ[i=1; N] (y,i - yHat,i)^2
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
    
    Args:
        yHat (Tensor) : estimations
        y (Tensor) : target values
    Returns:
        loss (Tensor) : 1/N * Σ[i=1; N] |y,i - yHat,i|
    '''
    loss = (y-yHat).abs().mean()

    return loss
