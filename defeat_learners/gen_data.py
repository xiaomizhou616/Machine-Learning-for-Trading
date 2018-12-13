"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees
def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    X = np.random.rand(300, 6) * 82. - 41.
    Y = X[:,0] * 21. + X[:,1] * 8. + X[:,2] * 3. + X[:,3] * 2. - X[:,4] * 7. + X[:,5] * 9.
    Y -= 3.
    
    return X, Y

def best4DT(seed=1489683273):
    np.random.seed(seed)
    X = np.random.rand(300, 6) * 82. - 41.
    Y = np.sin(3. * X[:,0]) + 2. * X[:,1]**2 + X[:,2]**3 + X[:,3]**2 + 3. * X[:,4]**3 + 4. * X[:,5]**2
    return X, Y

def author():
    return 'xhan306' #Change this to your user ID

if __name__=="__main__":
    print "they call me Tim."
