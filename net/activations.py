import numpy as np
from sklearn.utils.extmath import softmax as _softmax

def sigmoid(x,deriv = False):
    sig = 1 / (1 + np.exp(-x))
    if deriv:
        return sig*(1-sig)
    return sig

def relu(x,deriv=False):

    if deriv:
        x_deriv = x.copy()
        x_deriv[np.where(x > 0)] = 1
        x_deriv[np.where(x <= 0)] = 0
        return x_deriv
    return np.maximum(x,0)

def tanh(x,deriv=False):
    out = np.tanh(x)
    if deriv:
        return (1 - (out**2))
    return out

def softmax(X):
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X

if __name__ == "__main__":
    print (" ____ Test ______ ")
