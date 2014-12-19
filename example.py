import numpy as np


from NNet.layer import FullLayer, BiasLayer, ApplyFunctionLayer, OutputLayerMSE, ScaleLayer, SineLayer, SoftmaxLayer, OutputLayerNLL, ProjectionLayer
from NNet.nn import NN
from NNet.preprocess import preprocess, min_max_scaler, mean_std_scaler
from NNet.funcs import *
from NNet.learner import Learner
from NNet.utils import shuffle, divide, to_hamming, from_hamming, get_classes
from NNet.easy import EnsembleModel
from NNet.errors import confusion_matrix, classification_error
from NNet import rnd_gen

import re
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll import scope
from hyperopt.pyll.stochastic import sample

from collections import defaultdict

normal_ratios = [0.8, 0.2]
cross_validation_ratios = [0.8, 0.1, 0.1]
bootstrap_ratios = [0.8, 0.2]
presence, absence = 1, -1
max_evals_cv = 50
funcs = {
        "tanh": (tanh, d_tanh),
        "relu": (relu, d_relu),
        "sigmoid": (sigmoid, d_sigmoid),
        "id" : (lambda x:x, lambda x:np.ones(x.shape)),
}
config = {
    "hidden" : [30, 5],
    "activations": ["relu", "relu", "tanh"],
    "rnd" : rnd_gen.np_normal,
    "nb_iter": 800,
    "alpha":0.08,
    "momentum": 0.5
}

import json
def gen_spearmint_config(config):

    cfg = {}

    vars = {}
    for k, v in config.items():
        if type(v) == list:
            if type(v[0])==int:
                t="int"
                min=0
                max=20
            elif type(v[0])==float:
                t="float"
                min=0
                max=1
            size = len(v)
        elif type(v) == float:
            t="float"
            size=1
            min=0
            max=1
        elif type(v)==int:
            t="int"
            size=1
            min=0
            max=50
        t=t.upper()
        vars[k] = {"type":t,"size":size,"min":min,"max":max}


from itertools import repeat, chain
import collections
def build_layers(x_size, y_size, config):
    hidden = config.get("hidden")
    activations = config.get("activations")
    rnd_gen = config.get("rnd")
    
    if not isinstance(hidden, collections.Iterable):
        hidden = [hidden]
    if not isinstance(activations, collections.Iterable):
        activations = repeat(activations)
    if not isinstance(rnd_gen, collections.Iterable):
        rnd_gen = repeat(rnd_gen)
    

    layers = []
    i_size = x_size
    for h, a, r in zip( chain(hidden, [y_size]), activations, rnd_gen):
        o_size = h
        layers.append( FullLayer(i_size, o_size, r) )
        layers.append( BiasLayer(o_size) )
        f, d_f  = funcs[a]
        layers.append( ApplyFunctionLayer(f, d_f) )

        i_size = o_size
    
    layers.append( OutputLayerMSE() )
    #layers.append(  SoftmaxLayer() )
    #layers.append(  OutputLayerNLL())

    return layers

def gen_each_iter_classification_error(label, x, y, classes):


    y = np.array(from_hamming(y, classes))
    def f(learner):
        pred_y =  np.array(from_hamming( learner.nn.forward(x), classes ))
        return (label, classification_error( pred_y, y ) )
 
    return f


def gen_each_iter_confusion_matrix(label, x, y, classes):

    y = np.array(from_hamming(y, classes))
    def f(learner):
        pred_y =  np.array(from_hamming( learner.nn.forward(x), classes ))
        return (label, confusion_matrix( pred_y, y, classes) )
 
    return f



def gen_each_iter_get_loss(label, x, y):
    
    def f(learner):
        return label, learner.get_loss(x, y)
    
    return f

def gen_each_iter_regression_error(label, x, y):

    def f(learner):
        return (label, np.mean( np.abs ((learner.nn.forward(x)-y) / (y + (y==0) )) )  )
 
    return f


def build_model(x, y, config, build_layers=build_layers, each_iter=None):
    if each_iter is None:
        each_iter =  []


    layers = build_layers(x.shape[1], y.shape[1], config)

    alpha = config.get("alpha", 0.001)
    momentum = config.get("momentum", 0)
    lambda_ = config.get("lambda", 0)
    eta = config.get("eta", 0)

    nn = NN(layers)
    learner = Learner(nn, alpha=alpha, momentum=momentum, lambda_=lambda_, eta=eta)
    batch_size = config.get("batch_size", x.shape[0])
    nb_iter = config.get("nb_iter", 1000)
    report_loss_each = config.get("report_loss_each", 10)
    report_data_each = config.get("report_data_each", 50)
    
    data = []
    for i in xrange(nb_iter):
        learner.stochastic_gradient_descent(x, y, batch_size, it=i)
        datum = dict(fct(learner) for fct in each_iter) 
        data.append( datum )

        if i % report_loss_each == 0:
            print "iteration : %d" % (i,)
            print learner.get_loss(x, y)
        if i % report_data_each == 0:
            for k, v in datum.items():
                print k + ":"
                print v
    return nn, data



def gen_hp_optimize(x, y,  loss_getter=lambda data:0, *args, **kwargs):

    def f(config):
        print "Trying %s" % (config,)

        #x_, y_ = shuffle(x, y)
        x_ = x
        y_ = y

        nn, data = build_model(x_, y_, config, *args, **kwargs)
        return {"loss": loss_getter(data[-1]), "config": config, "data": data, "nn": nn}
    
    return f

def ensemble_model_cv(x, y, nb_models, *args, **kwargs):
    rt = 1. / nb_models
    rt_last = 1. - rt * (nb_models - 1)
    ratios = [rt] * (nb_models - 1) + [rt_last]
    
    divs = divide(x, y, ratios)
    models = []
    models_data = []
    for x_, y_ in divs:
        nn, data = build_model(x_, y_, *args, **kwargs)
        models.append(nn)
        models_data.append(data)
    ensemble = EnsembleModel(models)
    return ensemble, models_data

def ensemble_model(nb_models, *args, **kwargs):
    models = []
    models_data = []
    for i in xrange(nb_models):
        nn, data = build_model(*args, **kwargs)
        models.append(nn)
        models_data.append(data)
    ensemble = EnsembleModel(models)
    return ensemble, models_data

import itertools
def ravel(l):
    return list(itertools.chain.from_iterable(l))

normal, cross_validation, bootstrap = range(3) 
class Problem(object):

    def __init__(self, example, mode=normal,id=""):
        x, y = example
        self.id = id
        x, y = shuffle(x, y)
        self.x = x
        self.y = y
        self.mode = mode

    def __getstate__(self):
        d  = {}
        d["shape"] = (self.x.shape, self.y.shape)
        if self.mode == cross_validation:
            d.update( {"trials": self.trials} )
        elif self.mode == normal:
            d.update( {"nn_": self.nn_, "data_" : self.data_, "config":self.config_} )
        elif self.mode == bootstrap:
            d.update( {"nnlist": self.nnlist, "datalist": self.datalist, "configlist":self.configlist} )
        return d

    def pre(self):
        preprocessing_funcs = [mean_std_scaler]
        self.x = preprocess(self.x, preprocessing_funcs)

        if self.mode == cross_validation:
            (self.train_x, self.train_y), (self.valid_x, self.valid_y), (self.test_x, self.test_y) = divide(self.x, self.y, cross_validation_ratios)
        elif self.mode == normal:
            (self.train_x, self.train_y), (self.test_x, self.test_y) = divide(self.x, self.y, normal_ratios)
        elif self.mode == bootstrap:
            self.nnlist = []
            self.datalist = []
            self.configlist = []
            (self.train_x, self.train_y), (self.test_x, self.test_y) = divide(self.x, self.y, bootstrap_ratios)

        self.config = config
        self.validation_criterion = "loss"
 
    def learn(self):

        if self.mode == cross_validation:
            space = self.config
            trials = Trials()
            
            fmin(fn=gen_hp_optimize(self.train_x, self.train_y, 
                                    lambda data:data[self.validation_criterion], each_iter=self.each_iter), 
                                    space=space, algo=tpe.suggest, max_evals=max_evals_cv, trials=trials)
            self.trials = trials
        elif self.mode == normal:
            learn = gen_hp_optimize(self.train_x, self.train_y, 
                                lambda data:data[self.validation_criterion], each_iter=self.each_iter)
            cfg = sample(self.config)
            self.config_ = cfg
            res = learn(cfg)
            self.nn_, self.data_ = res["nn"], res["data"]

        elif self.mode == bootstrap:
            N = 10
            parts = divide(self.train_x, self.train_y, [1./N]*N)

            for i in xrange(N):
                parts_train = parts[0:i]+parts[i+1:]
                part_test = parts[i]

                train_x = np.array(ravel([x_ for x_, y_ in parts_train]))
                train_y = np.array(ravel([y_ for x_, y_ in parts_train]))
                test_x, test_y = part_test
                
                each_iter = self.get_each_iter(train_x, train_y, test_x, test_y)
                learn = gen_hp_optimize(train_x, train_y, 
                                        lambda data:data[self.validation_criterion], each_iter=each_iter)
                cfg = sample(self.config)
                self.configlist.append(cfg)
                res = learn(cfg)
                nn, data = res["nn"], res["data"]
                self.nnlist.append(nn)
                self.datalist.append(data)
            self.bootstrap_ensemble = EnsembleModel(self.nnlist)
    
    def show_result(self, raw_pred_y, y):
        pass

    def post(self):

        if self.mode == cross_validation:
            best_result = min(self.trials.trials, key=lambda trial:trial["result"]["loss"])["result"]

            print "Best caracs....."
            print best_result["config"]
            for k, v in best_result["data"][-1].items():
                print "Best caracs " + k+":"
                print v
            print
            models = [   trial["result"]["nn"]  for trial in self.trials.trials]
            ensemble  = EnsembleModel(models)
            print "Results ON TEST......"

            print "Ensemble:"
            self.show_result(ensemble.forward(self.test_x), self.test_y)
            print "Best one"
            nn = best_result["nn"]
            

            self.show_result(nn.forward(self.test_x), self.test_y) 
        elif self.mode == normal:
            print "train results:"
            self.show_result(self.nn_.forward(self.train_x), self.train_y)
            print "test results:"
            self.show_result(self.nn_.forward(self.test_x), self.test_y)
       
        elif self.mode == bootstrap:

            for l in ("train", "test"):
                errors = [data[-1][l] for data in self.datalist]
                print "%s results" % (l,)
                print errors
                print "Mean : %f" % (np.mean(errors),)
                print "Std : %f" % (np.std(errors),)

            print "using ensemble : "
            self.show_result(self.bootstrap_ensemble.forward(self.test_x), self.test_y)


class ClassificationProblem(Problem):

    def get_each_iter(self, train_x, train_y, test_x, test_y):
        return [gen_each_iter_classification_error("train", train_x, train_y, self.classes),
                gen_each_iter_classification_error("test", test_x, test_y, self.classes),
                gen_each_iter_confusion_matrix("train_cm", train_x, train_y, self.classes),
                gen_each_iter_confusion_matrix("test_cm", test_x, test_y, self.classes),
                gen_each_iter_get_loss("loss", train_x, train_y)]


    def pre(self):

        self.classes = get_classes(self.y[:, 0])
        self.y = to_hamming(self.y[:, 0], presence=presence, absence=absence)
        super(ClassificationProblem, self).pre()
        
        
        if self.mode == cross_validation:
            tx, ty = self.valid_x, self.valid_y
        elif self.mode == normal:
            tx, ty = self.test_x, self.test_y

        if self.mode in (cross_validation, normal):
            self.each_iter = self.get_each_iter(self.train_x, self.train_y, tx, ty)
            self.validation_criterion = "test"
    
    def show_result(self, raw_pred_y, y):
        pred_y =  from_hamming( raw_pred_y, self.classes )
        y = from_hamming(y, self.classes)
        print classification_error(pred_y, y)
        print confusion_matrix(pred_y, y, self.classes)

    def plot(self):
        if self.mode == normal:
            train = [d["train"] for d in self.data_]
            test = [d["test"] for d in self.data_]

            plt.plot(train, label="train")
            plt.plot(test, label="test")
            plt.legend()
            plt.savefig('img/classification-normal-%s.png' % (self.id,))

        elif self.mode == bootstrap:
            train = [d[-1]["train"] for d in self.datalist]
            test = [d[-1]["test"] for d in self.datalist]
            plt.xlabel("train")
            plt.ylabel("test")
            plt.scatter(train, test)
            plt.savefig('img/classification-bootstrap-%s.png' % (self.id,))
        elif self.mode == cross_validation:
            train = [   trial["result"]["data"][-1]["train"]  for trial in self.trials.trials ]
            test = [   trial["result"]["data"][-1]["test"]  for trial in self.trials.trials ]
            plt.xlabel("train")
            plt.ylabel("test")
            plt.scatter(train, test)
            plt.savefig('img/classification-cv-%s.png' % (self.id,))
 


class RegressionProblem(Problem):

    def get_each_iter(self, train_x, train_y, test_x, test_y):
        return [
            gen_each_iter_regression_error("train", train_x, train_y),
            gen_each_iter_regression_error("test", test_x, test_y),
            gen_each_iter_get_loss("loss", train_x, train_y),
        ]

    def pre(self):

        preprocessing_funcs = [min_max_scaler]
        self.y = preprocess(self.y, preprocessing_funcs)

        super(RegressionProblem, self).pre()
        
        if self.mode == cross_validation:
            tx, ty = self.valid_x, self.valid_y
        elif self.mode == normal:
            tx, ty = self.test_x, self.test_y

        if self.mode in (cross_validation, normal):
            self.each_iter = self.get_each_iter(self.train_x, self.train_y, tx, ty)
            self.validation_criterion = "loss"

    def show_result(self, raw_pred_y, y):
        print np.mean( np.abs(raw_pred_y - y) / np.abs(y + (y==0)) )

    def plot(self):
        if self.mode == normal:
            train = [d["train"] for d in self.data_]
            test = [d["test"] for d in self.data_]

            plt.plot(train, label="train mean abs rel error")
            plt.plot(test, label="test mean abs rel error")
            plt.legend()
            plt.savefig('img/regression-normal-%s.png'%(self.id,))
        elif self.mode == bootstrap:
            train = [d[-1]["train"] for d in self.datalist]
            test = [d[-1]["test"] for d in self.datalist]
            plt.xlabel("train mean abs rel error")
            plt.ylabel("test mean abs rel error")
            plt.scatter(train, test)
            plt.savefig('img/regression-bootstrap-%s.png' % (self.id,))
        elif self.mode == cross_validation:
            train = [   trial["result"]["data"][-1]["train"]  for trial in self.trials.trials ]
            test = [   trial["result"]["data"][-1]["test"]  for trial in self.trials.trials ]
            plt.xlabel("train")
            plt.ylabel("test")
            plt.scatter(train, test)
            plt.savefig('img/regression-cv-%s.png' % (self.id,))
 

import matplotlib as mpl
mpl.use('Agg')


import matplotlib.pyplot as plt
import md5
import cPickle as pickle

import copy_reg
from types import FunctionType
def stub_pickler(obj):
    return stub_unpickler, ()
def stub_unpickler():
    return "STUB"
copy_reg.pickle(FunctionType, stub_pickler, stub_unpickler)



from datasets import mnist_mimic_example, cifar_example
def cifar():
    models_id = open("all_cifar_models", "r").readlines()
    models_id = map(lambda x:x.replace("\n", ""), models_id)

    models = []
    for model_id in models_id:
        model_data  = pickle.load(open("res/problem-%s.data" % ( model_id ) ))
        model = model_data.nn_
        models.append(model)
    ens_model = EnsembleModel(models)
    
    cifar_data = cifar_example(6) # 6 = test data
    x, y = cifar_data
    print classification_error(np.argmax(ens_model.forward(x), axis=1), y[:, 0])
    sys.exit(0)
    
import datasets
import sys

def run(argv):
    examples = getattr(datasets,argv[1])(*argv[2:])
    import time
    begin = time.time()
    id = md5.new( str(time.time()) ).hexdigest()
    problem = ClassificationProblem(examples , mode=normal, id=id)
    #problem = RegressionProblem(examples, mode=normal, id=id)
    problem.pre()
    problem.learn()
    problem.post()
    problem.plot()
    print
    print
    print id 
    pickle.dump(problem, open("res/problem-%s.data" % (id,), "w"))
    print "It took %f sec" % (time.time() - begin,)
if __name__ == "__main__":
    run(sys.argv)
