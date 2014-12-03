import numpy as np


def uniform(state, size=(1,)):
    return state.uniform(-2. / np.prod(size), 2. / np.prod(size), size=size)

def normal(state, size=(1,)):
    return state.normal(0, 1. / np.sqrt(np.prod(size)), size=size)

def zeros(state, size=(1,)):
    return np.zeros(size)


def gen_func(state, orig_func):

    def f(size):
        return orig_func(state, size)
    f.__name__ = orig_func.__name__
    return f

np_normal = gen_func(np.random, normal)
np_uniform = gen_func(np.random, uniform)
