from utils import get_batches, evolving_rate
import numpy as np

momentum, adadelta, nag = range(3)
class Learner(object):
    
    def __init__(self, nn, alpha=0.01, momentum=0.9, lambda_=1, eta=1, rho=0.95, eps=0.00001, method=adadelta, dropout=None):
        self.nn = nn
        self.rho = rho
        self.eps = eps
        self.alpha = alpha
        self.eta = eta
        self.momentum = momentum
        self.lambda_ = lambda_
        self.velocity = [0] * len(self.nn.layers)
        self.dx_squared = [0] * len(self.nn.layers)
        self.g_squared = [0] * len(self.nn.layers)
        self.dropout = dropout

        self.method = method
    
    def predict(self, x):
        if self.dropout is None:
            return self.nn.forward(x)
        else:
            for i, layer in enumerate(self.nn.layers):
                x *= (1-self.dropout[i])
                x = layer.forward(x)
            x *= (1-self.dropout[-1])
            return x


    def stochastic_gradient_descent(self, x, y, batch_size, it=0):
        #alpha = self.alpha
        #momentum = self.momentum
       
        alpha = evolving_rate(self.alpha, self.eta, it)
        momentum = evolving_rate(self.momentum, self.eta, it)
        lambda_ = self.lambda_

        for x_, y_ in get_batches(x, y, batch_size):
            o = self.nn.forward_get_all_outputs(x_)
            if self.dropout is not None:
                for i, p in enumerate(self.dropout[1:]):
                    o[i] = o[i] * (1-(np.random.uniform(size=o[i].shape) <= p))

            dweights = self.nn.get_dweights(x_, o, y_)
            #dweights_ = self.nn.estimate_dweights(x_, y_)
            #max_error = max( np.max( np.abs(a-b)/ (np.abs(b)+(b==0))  ) for a, b in zip(dweights, dweights_) if a is not None and b is not None )
            #print "max_error : %f" % (max_error,)
            #print np.column_stack( (dweights[1], dweights_[1]) )

            for (v, vel), dweight, layer in zip(enumerate(self.velocity), dweights, self.nn.layers):
                #if vel is None:vel = 0
                #if dweight is None:dweight = 0
                if hasattr(layer, "W"):
                    if layer.__class__.__dict__.get("regularizable", True):
                        reg = 2. * lambda_ * layer.W
                    else:
                        reg = 0.
                    g = dweight + reg
                    # momentum:
                    if self.method == momentum:
                        w = -alpha * g
                        self.velocity[v] = self.velocity[v]* momentum + (1 - momentum) * w

                    # adadelta
                    if self.method == adadelta:
                        self.g_squared[v] =   self.rho * self.g_squared[v] + (1 - self.rho) * (g**2)
                        rms_dx = np.sqrt(self.dx_squared[v] + self.eps)
                        self.dx_squared[v] =  self.rho * self.dx_squared[v] + (1 - self.rho) * self.velocity[v]**2
                        rms_g = np.sqrt(self.g_squared[v] + self.eps)
                        self.velocity[v] = - alpha * (rms_dx / rms_g) * g

            self.nn.gradient_ascent(self.velocity)

    def get_loss(self, x, y):
        reg = 0.
        for layer in self.nn.layers:
            if hasattr(layer, "W"):
                if layer.__class__.__dict__.get("regularizable", True):
                    reg += np.sum(layer.W**2)
        loss = self.nn.get_loss(x, y) + self.lambda_ * reg
        return loss
