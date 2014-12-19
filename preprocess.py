import numpy as np
def mean_std_scaler(x):
    return (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + (np.std(x, axis=0)==0) )

def min_max_scaler(x):
    min_ = np.min(x, axis=0)
    max_ = np.max(x, axis=0)
    return (x - min_) / (max_ - min_ + (max_==min_)  )

def preprocess(x, funcs):
    for f in funcs:
        x = f(x)
    return x


