import numpy as np
from NN import *

default_setting = {
    'input_dim': 784,
    'layers': [(100, sigmoid), (784, sigmoid)],
    'random_seed': 1,
    'weights_left_limit': -0.1,
    'weights_right_limit': 0.1,
    'bias_init': 0.01,
    'batch_size': 30,
    'learning_rate': 0.1,
}

class Autoencoder(NN):
    def __init__(self, setting):
        if not setting:
            setting = default_setting
        super(Autoencoder, self).__init__(setting)

    def train(self, X_train, Y_train, X_validate, Y_validate, cost_function=sigmoid_cross_entropy_loss,
              evaluate_function=sigmoid_cross_entropy_loss, n_epoch=50, hidden_layer_dropout=0.1, input_layer_dropout=0.1):
        return self._train(X_train, Y_train, X_validate, Y_validate, cost_function=cost_function, evaluate_function=evaluate_function, n_epoch=n_epoch, hidden_layer_dropout=hidden_layer_dropout, input_layer_dropout=input_layer_dropout)
