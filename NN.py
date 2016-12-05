import numpy as np
from matplotlib import pyplot as plt
import pickle
from evaluation_functions import *
from utils import dropout
import collections

def sigmoid(x, derivative=False):
    x = np.clip(x, -500, 500)
    x = 1.0 / (1.0 + np.exp(-x))
    if derivative:
        return np.multiply(x, 1-x)
    else:
        return x

def sigmoid_cross_entropy_loss(output, target, derivative=False):
    assert output.shape == target.shape, '[Error] output shape should be the same as target shape'
    if derivative:
        return output - target
    else:
        n_samples = output.shape[1]
        loss = -np.sum(np.sum(target * np.log(output) + (1 - target) * np.log(1 - output), axis=0), axis=0) / float(
            n_samples * output.shape[0])
        return loss

def softmax(x, derivative=False):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    x = e / np.sum(e, axis=1, keepdims=True)
    if derivative:
        return np.ones(x.shape)
    else:
        return x

def softmax_cross_entropy_loss(output, target, derivative=False, epsilon=1e-11):
    assert output.shape == target.shape, '[Error] output shape should be the same as target shape'
    output = np.clip(output, epsilon, 1-epsilon)
    if derivative:
        return target - output
    else:
        return np.mean(-np.sum(target * (np.log(output)), axis=0))

def cross_entropy_loss(output, target):
    assert output.shape == target.shape, '[Error] output shape should be the same as target shape'
    n_samples = output.shape[1]
    loss = -np.sum(np.sum(target * np.log(output) + (1 - target) * np.log(1 - output), axis=0), axis=0) / float(n_samples * output.shape[0])
    return loss


default_setting = {
    'random_seed': 1,
    'weights_left_limit': -0.1,
    'weights_right_limit': 0.1,
    'bias_init': 0.01,
    'batch_size': 30,
    'learning_rate': 0.1,
    'backprop_type': 'Vanilla',
    'momentum_factor': 0.1,
    'decay_rate': 0.99,
    'decay_rate_epsilon': 1e-8,
    'adam_beta1': 0.9,
    'adam_beta2': 0.9,
    'adam_epsilon': 1e-8,
}

backprop_types = {'Adam', 'Vanilla', 'RMSprop', 'Momentum'}

class NN(object):
    def __init__(self, setting):
        self.__dict__.update(default_setting)
        self.__dict__.update(setting)
        np.random.seed(self.random_seed)
        self.n_weights = (self.input_dim+1)*self.layers[0][0] + sum((self.layers[i][0]+1)*self.layers[i+1][0] for i in range(len(self.layers)-1))
        weights_array = np.random.uniform(self.weights_left_limit, self.weights_right_limit, size=(self.n_weights))
        self.set_weights(weights_array)
        self.output_dim = self.layers[-1][0]
        for i in range(len(self.layers)):
            self.weights[i][:, -1] = self.bias_init
        self.train_error = None
        self.validate_error = None
        assert self.backprop_type in backprop_types, '[Error] backprop_type should be Adom, Vanilla, RMSprop or Momentum'
        if self.backprop_type == 'Adam':
            self.m = [np.zeros(shape=weight_layer.shape) for weight_layer in self.weights]
            self.v = [np.zeros(shape=weight_layer.shape) for weight_layer in self.weights]
        elif self.backprop_type == 'RMSprop' or self.backprop_type == 'Momentum':
            self.cache = collections.defaultdict(int)

    def set_weights(self, weights_array):
        assert weights_array.shape[0] == self.n_weights, '[Error] weights_array shape is wrong'
        self.weights = []
        start_ind = 0
        for i, layer in enumerate(self.layers):
            if i == 0:
                n_weights = layer[0] * (self.input_dim + 1)
                self.weights.append(weights_array[start_ind:start_ind + n_weights].reshape((layer[0], self.input_dim + 1)))
            else:
                n_weights = layer[0] * (self.layers[i - 1][0] + 1)
                self.weights.append(weights_array[start_ind:start_ind + n_weights].reshape((layer[0], self.layers[i - 1][0] + 1)))
            start_ind += n_weights

    def set_first_layer_weights(self, weight_array):
        assert weight_array.shape[0] == self.layers[0][0] * (self.input_dim+1), '[Error] weight_array shape is wrong'
        self.weights[0] = weight_array.reshape(self.layers[0][0], self.input_dim+1)

    def get_weights_array(self):
        weights_array = np.hstack(np.array(weight).flat for weight in self.weights)
        assert weights_array.shape[0] == self.n_weights, '[Error] weights_array shape is wrong'
        return weights_array

    def get_first_layer_weights(self):
        weight_array = np.array(self.weights[0].flat)
        return weight_array

    def forward(self, X, trace=False):
        assert X.shape[1] == self.input_dim, '[Error] X shape is wrong'
        n_samples = X.shape[0]
        if trace:
            outputs, derivatives = [X.T], []
        for i, layer in enumerate(self.layers):
            if i == 0:
                output = np.dot(self.weights[i][:, :-1], X.T) + self.weights[i][:, -1].reshape(layer[0], 1)
            else:
                output = np.dot(self.weights[i][:, :-1], output) + self.weights[i][:,-1].reshape(layer[0], 1)
            output = layer[1](output)
            assert output.shape == (layer[0], n_samples), '[Error] output shape is wrong'
            if trace:
                outputs.append(output)
                derivative = layer[1](output, derivative=True)
                assert derivative.shape == (layer[0], n_samples), '[Error] derivative shape is wrong'
                derivatives.append(derivative)
        if trace:
            return outputs, derivatives
        else:
            return output

    def backpropagate(self, X, Y, cost_function, hidden_layer_dropout=0.1, input_layer_dropout=0.1):
        assert X.shape == (self.batch_size, self.input_dim), '[Error] X shape is wrong'
        assert Y.shape == (self.batch_size, self.output_dim), '[Error] Y shape is wrong'
        outputs, derivatives = self.forward(X, trace=True)
        output = outputs[-1]
        cost_derivative = cost_function(output, Y.T, derivative=True)
        da = cost_derivative
        for k in range(len(self.layers)-1, -1, -1):
            outputs[k] = dropout(outputs[k], hidden_layer_dropout if k > 0 else input_layer_dropout)
            assert da.shape == (self.layers[k][0], self.batch_size), '[Error] da shape is wrong'
            dW = np.dot(da, outputs[k].T) / float(self.batch_size)
            db = (np.sum(da, axis=1) / float(self.batch_size)).reshape(self.layers[k][0], 1)
            dW = np.hstack((dW, db))
            if k > 0:
                dh = (np.sum(np.dot(self.weights[k][:, :-1].T, da), axis=1) / float(self.batch_size)).reshape(self.layers[k-1][0], 1)
                da = dh * derivatives[k-1]
            dW = self.backpropagation_type(k, dW)
            self.weights[k] += dW

    def backpropagation_type(self, layer_index, dW):
        if self.backprop_type == 'Vanilla':
            return -self.learning_rate * dW
        elif self.backprop_type == 'Adam':
            self.m[layer_index] = self.adam_beta1 * self.m[layer_index] + (1 - self.adam_beta1) * dW
            self.v[layer_index] = self.adam_beta2 * self.v[layer_index] + (1 - self.adam_beta2) * (dW ** 2)
            return -self.learning_rate * self.m[layer_index] / (np.sqrt(self.v[layer_index]) + self.adam_epsilon)
        elif self.backprop_type == 'RMSprop':
            self.cache[layer_index] = self.decay_rate * self.cache[layer_index] + (1 - self.decay_rate) * dW ** 2
            return -self.learning_rate * dW / (np.sqrt(self.cache[layer_index]) + self.decay_rate_epsilon)
        elif self.backprop_type == 'Momentum':
            dW = -self.learning_rate * dW + self.momentum_factor * self.cache[layer_index]
            self.cache[layer_index] = dW
            return dW

    def train(self, X_train, Y_train, X_validate, Y_validate, cost_function, evaluate_function, n_epoch=50, hidden_layer_dropout=0.1, input_layer_dropout=0.1):
        return self._train(X_train, Y_train, X_validate, Y_validate, cost_function, evaluate_function, n_epoch=n_epoch, hidden_layer_dropout=hidden_layer_dropout, input_layer_dropout=input_layer_dropout)

    def _train(self, X_train, Y_train, X_validate, Y_validate, cost_function, evaluate_function, n_epoch=50, hidden_layer_dropout=0.1, input_layer_dropout=0.1):
        assert X_train.shape[1] == self.input_dim and Y_train.shape[1] == self.output_dim and X_train.shape[0] == Y_train.shape[0], \
            '[Error] X shape or Y shape is wrong'
        n_samples = X_train.shape[0]
        self.train_error = np.zeros((n_epoch, 1))
        self.validate_error = np.zeros((n_epoch, 1))
        for e in range(n_epoch):
            start_ind = 0
            for i in range(n_samples/self.batch_size):
                self.backpropagate(X_train[start_ind:start_ind+self.batch_size], Y_train[start_ind:start_ind+self.batch_size],
                                   cost_function=cost_function, hidden_layer_dropout=hidden_layer_dropout, input_layer_dropout=input_layer_dropout)
                start_ind += self.batch_size
            train_output = self.forward(X_train)
            validate_output = self.forward(X_validate)
            self.train_error[e] = evaluate_function(train_output, Y_train.T, derivative=False)
            self.validate_error[e] = evaluate_function(validate_output, Y_validate.T, derivative=False)
            if evaluate_function == classification_accuracy:
                print('[Train] epoch {0}, train accuracy is {1}, validate accuracy is {2}'.format(e, self.train_error[e], self.validate_error[e]))
            else:
                print('[Train] epoch {0}, train error is {1}, validate error is {2}'.format(e, self.train_error[e], self.validate_error[e]))
        return self.train_error, self.validate_error

    def plot_W(self, figname):
        weight = self.weights[0][:, :-1]
        assert weight.shape == (self.layers[0][0], self.input_dim), '[Error] weight shape is wrong'
        f, axarr = plt.subplots(10, 10, sharex='col', sharey='row')
        for i in range(self.layers[0][0]):
            r = i / 10
            c = i % 10
            w = weight[i].reshape(28, 28)
            axarr[r, c].imshow(w, cmap='gray')
        plt.show()
        f.savefig('{0}.pdf'.format(figname))

    def save_model(self, name):
        pickle.dump(self, open('{0}.pkl'.format(name), 'wb'))

    def load_model(self, name):
        model = pickle.load(open(name, 'rb'))
        return model