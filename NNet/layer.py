import numpy as np
from funcs import *
from utils import as_one_dim_array, safe_exp, safe_log
import utils
from itertools import product
import rnd_gen

class Layer(object):

    def forward(self, x):
        pass

    def backward(self, dx):
        pass


class SoftmaxLayer(object):
    def __init__(self):
        pass

    def forward(self, x):
        max_ = np.max(x, axis=1)[:, np.newaxis]
        exp_x = safe_exp(x - max_)
        sum_exp_x = np.sum(exp_x, axis=1)[:, np.newaxis]
        return exp_x / sum_exp_x

    def backward_with_output(self, x, dx, o):
        s_ = np.sum(dx * o, axis=1)[:, np.newaxis]
        return o * (  dx - s_  )

def_rnd_gen_full = rnd_gen.gen_func(np.random, rnd_gen.normal)



class FullLayer(object):

    def __init__(self, nb_input, nb_output, rnd_gen=def_rnd_gen_full):
        self.W = rnd_gen(size=(nb_output, nb_input))

    def forward(self, x):
        return as_one_dim_array(np.dot( x, self.W.T )  )

    def backward(self, x, dx):
        return as_one_dim_array(np.dot(dx, self.W ))

def_rnd_gen_bias = rnd_gen.gen_func(np.random, rnd_gen.zeros)
class BiasLayer(object):
    regularizable = False
    inplace = True
    def __init__(self, nb_input, rnd_gen=def_rnd_gen_bias):
        self.W = rnd_gen(size=(nb_input, 1))

    def forward(self, x):
        return x + self.W.T

    def backward(self, x, dx):
        return dx

class ScaleLayer(object):
    regularizable = False
    inplace = True
    def __init__(self, nb_input):
        self.W = np.ones(  (nb_input, 1) )

    def forward(self, x):
        return x * self.W.T

    def backward(self, x, dx):
        return dx * self.W.T


class OutputLayerMSE(object):

    def forward(self, x):
        return x

    def backward(self, x, dx):
        return dx

    def get_loss(self, outputs, expected_outputs):
        return np.sum((outputs - expected_outputs)**2)

    def get_output_d(self, outputs, expected_outputs):
        return 2. * (outputs - expected_outputs)


class OutputLayerMI(object):

    def forward(self, x):
        return x

    def backward(self, x, dx):
        return dx

    def get_loss(self, outputs, expected_outputs):
        return -np.sum(np.log(outputs) * expected_outputs + np.log(1 - outputs) * (1 - expected_outputs))

    def get_output_d(self, outputs, expected_outputs):
        return (-(expected_outputs / outputs) + (1 - expected_outputs) / (1 - outputs))



class OutputLayerNLL(object):

    def forward(self, x):
        return x

    def backward(self, x, dx):
        return dx

    def get_loss(self, outputs, expected_outputs):
        where_max = np.argmax(expected_outputs, axis=1)
        return -np.sum(safe_log(np.choose(where_max, outputs.T)))

    def get_output_d(self, outputs, expected_outputs):
        where_max = np.argmax(expected_outputs, axis=1)
        d_ = np.zeros( outputs.shape )
        d_[expected_outputs==1] = -1. / outputs[expected_outputs==1]
        return d_


class ApplyFunctionLayer(object):

    def __init__(self, f, d_f):
        self.f = f
        self.d_f = d_f

    def forward(self, x):
        return self.f(x)

    def backward(self, x, dx):
        return self.d_f(x) * dx

class SineLayer(object):
    inplace = True
    def __init__(self):
        self.W = np.array(np.ones([1, 1]))

    def forward(self, x):
        return np.sin(x)

    def backward(self, x, dx):
        return np.cos(x) * dx

class ProjectionLayer(object):

    def __init__(self, vocab_size, dim_projection, rnd_gen=def_rnd_gen_full):
        self.W = rnd_gen( size=(vocab_size, dim_projection) )


    def forward(self, x):
        r =  np.tensordot(x, self.W, axes=(2, 0))
        return r.reshape( r.shape[0], (r.shape[1]*r.shape[2])  )

    def backward(self, x, dx):
        pass
