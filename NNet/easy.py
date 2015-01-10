from utils import get_batches
import numpy as np
def cross_validation(model_builder, model_evaluator, x, y, ratios, nb_batches):
    (train_x, train_y), (valid_x, valid_y) = divide(x, y, ratios)

    nb_examples = x.shape[0]
    batch_size = nb_examples / nb_batches

    evals = []
    models = []
    for x_, y_ in get_batches(train_x, train_y, batch_size):
        nn = model_builder(x_, y_)
        evaluation = model_evaluator(nn, valid_x, valid_y)
        models_evaluations.append( (nn, evaluation) )
    return models_evaluations


class EnsembleModel(object):

    def __init__(self, models):
        self.models = models

    def predict(self, x):
        outputs = [model.predict(x) for model in self.models]
        return np.mean(outputs, axis=0)

class StackModel(object):

    def __init__(self, models):
        self.models = models

    def predict(self, x):
        for model in self.models:
            x = self.predict(x)
        return x


def get_best_model(models_evaluations):
    m, _ = max(models_evaluations, key=lambda (m, e):e)
    return m

def get_ensemble_model(models_evaluations):
    return EnsembleModel( map(lambda (m, e): m, models_evaluations) )
