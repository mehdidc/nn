import numpy as np

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return 1.*(x > 0)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))


def gauss(x):
    return np.exp(-(x**2)*0.5)


def d_gauss(x):
    return x * np.exp(-(x**2)*0.5)


def sin(x):
    return np.sin(x)

def d_sin(x):
    return np.cos(x)
