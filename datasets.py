import numpy as np
import re

def cifar_example(nb=1, nb_features=4096):
    nb=int(nb)
    nb_features=int(nb_features)
    if nb==7:
        per_batch = 10000
        all_data = np.zeros(   (per_batch * 6, nb_features) )
        all_labels = np.zeros(  (per_batch * 6, 1) )
        for n in (1, 2, 3, 4, 5, 6):
            data, labels = np.load("data/cifar%d_%d.npy" % (n, nb_features))
            all_data[(n - 1) * per_batch: n * per_batch] = np.array(list(data))
            all_labels[(n - 1) * per_batch : n * per_batch] = np.array([[l] for l in labels])
        return all_data, all_labels
    else:
        data, labels = np.load("data/cifar%d_%d.npy" % (nb, nb_features))
        data = np.array(list(data))
        labels = np.array([[l] for l in labels])
        data = data[0:200]
        labels = labels[0:200]
        return data, labels

def mnist_mimic_example():
    data, targets = np.load("data/mnist_new_pca.npy")
    data = np.array(list(data))
    data = data.reshape( (data.shape[0], np.prod(data.shape[1:])))
    targets = np.array(list(targets))
    #data = data[0:5000]
    #targets=targets[0:5000]
    return data, targets

def mnist_example():

    data, targets = np.load("data/mnist.npy")
    data = np.array(list(data))
    data = data.reshape( (data.shape[0], np.prod(data.shape[1:])))
    targets = np.array([ [int(d)] for d in targets])
    return data, targets


def sin_example():
    x = np.array([ [float(k)] for k in np.arange(-np.pi, np.pi, 0.01)])
    y = np.array([np.sin(k) for k in x])
    return x, y

def breast_example():
    dataset = np.loadtxt("data/breast.txt", delimiter=",")
    x, y = dataset[:, 0:-1], dataset[:, -1]
    y = np.array([[yi] for yi in y])
    return x, y

def plants_example():
    dataset = np.loadtxt("data/data_Mar_64.txt", delimiter=",", dtype="string")
    x, y = dataset[:, 1:].astype("float"), dataset[:, 0]
    y = np.array([[yi] for yi in y])
    return x, y


def page_blocks_example():
    dataset = np.loadtxt("data/page-blocks.data")
    x, y = dataset[:, 0:-1], dataset[:, -1]
    y = np.array([[yi] for yi in y])
    return x, y


def text_example(txt):

    txt = open(txt, "r").read()
    txt = re.split("[\. ]", txt)
    
    
    vocab = defaultdict(int)
    for t in txt:
        vocab[t] += 1
    del vocab['']
    limit_vocab = 80

    
    vocab = sorted(vocab.items(), key=lambda (k,v):v, reverse=True)[0:limit_vocab]
    vocab = map(lambda (a,b):a, vocab)
    
    vocab_repr = {}
    vocab_repr.update(dict((vocab[i],  [1 if r==i else -1 for r in xrange(len(vocab))]  ) for i in xrange(len(vocab))))

    window = 2
    x = []
    y = []
    for p in xrange(window, len(txt) - window):
        before = txt[p - window:p]
        after = txt[p + 1:p + window + 1]
        cur = txt[p]

        if cur in vocab_repr and all(w in vocab_repr for w in before + after):
            x_ = []

            for w in before + after:
                x_.extend(vocab_repr[w])
 
            y_ = vocab_repr[cur]
            
            x.append(x_)
            y.append( [np.argmax(vocab_repr[cur])] )
    return np.array(x), np.array(y)


def taxonomy_example(name):
    fd = open("data/taxonomy/%s" % (name,), "r")
    x = []
    y = []
    for l in fd.readlines():
        s = l.split(",")
        x.append(map(float,s[1:]))
        y.append([ s[0] ])

    fd.close()
    return np.array(x), np.array(y)
