import numpy as np

def log_transform(X, alpha, eps):
    def f(x):
        return 1+alpha*np.log(1+x/eps)
    Y = X.copy()
    Y.data = f(Y.data)
    return Y

def linear_transform(X, alpha):
    def f(x):
        return 1+alpha*x
    Y = X.copy()
    Y.data = f(Y.data)
    return Y