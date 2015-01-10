import numpy as np
import copy


class Stats(object):

    def __init__(self):
        self.data = defaultdict(list)

    def subjects(self):
        return self.data.keys()

    def new_point(self, subject, p):
        self.data[subject].append(p)

    def get_points(self, subject):
        return self.data[subject]


def safe_exp(x):
    return np.exp(x)

def safe_log(x):
    return np.log(x)

def as_one_dim_array(a):
    return a.reshape( (a.shape[0], np.prod(a.shape[1:]))   )


def get_classes(targets):
    return sorted(set(targets))

def to_hamming(targets, presence=1, absence=-1):
    t = sorted(set(targets))
    mapping = {}
    M = [absence] * len(t)
    for i, m in enumerate(t):
        M[i] = presence
        mapping[m] = copy.copy(M)
        M[i] = absence
    return np.array([mapping[t] for t in targets])

def from_hamming(targets, classes):
    return np.array(map( lambda x:classes[x], np.argmax(targets, axis=1) ))

def divide(x, y, ratios):

    datasets = []

    nb_examples = x.shape[0]
    first = 0

    for ratio in ratios:
        nb = int(nb_examples * ratio)
        last = first + nb
        datasets.append( (x[first:last], y[first:last]) )
        first = last
    return datasets

def get_batches(x, y, batch_size):
        nb_examples = x.shape[0]
        nb_chunks = nb_examples / batch_size
        if nb_examples % batch_size > 0:
            nb_chunks += 1

        first = 0
        for i in xrange(nb_chunks):
            last = min(first + batch_size, nb_examples)
            x_ = x[first:last]
            y_ = y[first:last]
            yield x_, y_

def shuffle(x, y):
    nb_examples = x.shape[0]
    k = range(nb_examples)
    np.random.shuffle(k)
    return x[k], y[k]

def evolving_rate(alpha0, lambda_, iteration):
    return alpha0  * (1. / (1 + alpha0 * lambda_ * iteration))
