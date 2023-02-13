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

if __name__ == '__main__':
    print(test_cross_entropy())
