from itertools import product
import numpy as np
import numexpr as ne
class NN(object):
    
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        return self.forward_until(x, len(self.layers))

    def forward_until(self, x, layer):
        for layer in self.layers[0:layer]:
            x = layer.forward(x)
        return x

    
    def forward_get_all_outputs(self, x):
        outputs = []
        for layer in self.layers:
            x = layer.forward(x)
            outputs.append(x)
        return outputs
    
    def get_dweights(self, inputs, outputs, expected_outputs):
        dweights = []
        dx = self.layers[-1].get_output_d(outputs[-1], expected_outputs)
        k = -2

        outputs = [inputs] + outputs
        for layer in reversed(self.layers):
            x = outputs[k]
            
            inplace = layer.__class__.__dict__.get("inplace", False)
            
            if inplace == False:
                dweights.append( np.sum((x[:, :, np.newaxis] *dx[:, np.newaxis, :]), axis=0).T ) 
            elif inplace == True:
                dweights.append( np.sum(dx, axis=0)[:, np.newaxis] )

            if hasattr(layer, "backward"):
                dx = layer.backward(x, dx)
            elif hasattr(layer, "backward_with_output"):
                o = outputs[k  + 1]
                dx = layer.backward_with_output(x, dx, o)
            k -= 1
        return list(reversed(dweights))
    
    def get_loss(self, x, y):
        return self.layers[-1].get_loss(self.forward(x), y)

    def estimate_dweights(self, inputs, expected_outputs, epsilon=10E-6):
        dweights = []
        for layer in self.layers:
            if hasattr(layer, "W"):
                W = np.zeros(layer.W.shape)
                
                indexes = [xrange(s) for s in layer.W.shape]
                
                for coords in (product(*indexes)):
                    old = layer.W[coords]
                    layer.W[coords] = old + epsilon
                    err_right = self.get_loss(inputs, expected_outputs)
                    layer.W[coords] = old - epsilon
                    err_left = self.get_loss(inputs, expected_outputs)
                    layer.W[coords] = old
                    W[coords] = (err_right - err_left) /  (2*epsilon)
                dweights.append(W)
            else:
                dweights.append(None)
        return dweights

    def gradient_ascent(self, dweights):
        for layer, dweight in zip(self.layers, dweights):
            if hasattr(layer, "W") :
                layer.W += dweight
