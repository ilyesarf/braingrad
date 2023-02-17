import sys
sys.path.insert(1, '..')
import numpy as np
from losses import *

def test_cross_entropy():
    y = np.array([0, 1, 0])
    yHat_1 = np.array([0.1, 0.8, 0.1])
    yHat_2 = np.array([0.1, 0.5, 0.1])

    cost_1 = cross_entropy(yHat_1, y)
    cost_2 = cross_entropy(yHat_2, y)
    
    assert cost_1 < cost_2

def test_mean_squared_error():
    y = np.array([1,2,5])
    yHat_1 = np.array([0.7, 1.8, 5.3])
    yHat_2 = np.array([0.3, 1, 4.5])

    mse_1 = mean_squared_error(yHat_1, y)
    mse_2 = mean_squared_error(yHat_2, y)

    assert mse_1 < mse_2

def test_mean_absoulte_error():
    y = np.array([1,2,5])
    yHat_1 = np.array([0.7, 1.8, 5.3])
    yHat_2 = np.array([0.3, 1, 4.5])

    mse_1 = mean_absolute_error(yHat_1, y)
    mse_2 = mean_absolute_error(yHat_2, y)

    assert mse_1 < mse_2

