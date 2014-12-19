import numpy as np
from utils import get_classes

def classification_error(y, realy):
    return np.mean(realy != y)

def confusion_matrix(y, realy, classes):
    nb_classes = len(classes)
    confmat = np.zeros((nb_classes, nb_classes))
    for i in xrange(nb_classes):
        for j in xrange(nb_classes):
            confmat[i, j] = np.sum( (realy == classes[i] ) * (y == classes[j]) )
    return confmat
