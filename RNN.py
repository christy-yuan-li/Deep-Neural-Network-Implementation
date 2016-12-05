import numpy as np
from matplotlib import pyplot as plt
import pickle
from evaluation_functions import *


def sigmoid(x, derivative=False):
    x = np.clip(x, -500, 500)
    x = 1.0 / (1.0 + np.exp(-x))
    if derivative:
        return np.multiply(x, 1-x)
    else:
        return x

def sigmoid_cross_entropy_loss_allseq(output, target, derivative=False):
    seq_len = len(output.keys())
    assert set(output.keys()) == set(target.keys()) == set(
        range(seq_len)), '[Error] prediction and targets should have same sequence length and keys'
    loss, der = 0, {}
    n_samples = output[0].shape[1]
    for t in range(seq_len):
        assert output[t].shape == target[t].shape, '[Error] output[{0}] and target[{0}] has different shape'.format(t)
        der[t] = output[t] - target[t]
        loss += -np.sum(np.sum(target[t] * np.log(output[t]) + (1 - target[t]) * np.log(1 - output[t]), axis=0), axis=0) / float(
            n_samples)
    if derivative:
        return der
    else:
        return float(loss) / float(seq_len)

def softmax(x, derivative=False):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    x = e / np.sum(e, axis=1, keepdims=True)
    if derivative:
        return np.ones(x.shape)
    else:
        return x

def softmax_cross_entropy_loss_allseq(output, target, derivative=False, epsilon=1e-11):
    seq_len = len(output.keys())
    assert set(output.keys()) == set(target.keys()) == set(range(seq_len)), '[Error] prediction and targets should have same sequence length ' \
                                                                      'and keys'
    loss, der = 0, {}
    for t in range(seq_len):
        assert output[t].shape == target[t].shape, '[Error] output[{0}] and target[{0}] has different shape'.format(t)
        # output[t] = np.clip(output[t], epsilon, 1 - epsilon)
        loss += np.mean(-np.sum(target[t] * (np.log(output[t])), axis=0))
        der[t] = target[t] - output[t]
        # der[t] = output[t] - target[t]
    if derivative:
        return der
    else:
        return loss / float(seq_len)


default_setting = {
    'input_dim': None,
    'layers': None,
    'seq_len': 10,
    'batch_size': 30,
    'learning_rate': 0.01,
    'cost_function': softmax_cross_entropy_loss_allseq,
    'random_seed': 1,
    'weights_left_limit': -0.1,
    'weights_right_limit': 0.1,
    'bias_init': 0.01,
    'backprop_type': 'Vanilla',
    'momentum_factor': 0.1,
    'decay_rate': 0.99,
    'decay_rate_epsilon': 1e-8,
    'adam_beta1': 0.9,
    'adam_beta2': 0.9,
    'adam_epsilon': 1e-8,
}


class RNN(object):
    def __init__(self, setting):
        self.__dict__.update(default_setting)
        self.__dict__.update(setting)
        np.random.seed(self.random_seed)
        self.train_error = None
        self.validate_error = None
        assert self.input_dim is not None and self.layers is not None, '[Error] input_dim and layers must be specified in initialization'
        assert (self.layers[1][1] == 'softmax' and self.cost_function == 'softmax_cross_entropy_loss') or self.layers[1][1] != 'softmax', \
            '[Error] if last layer activation is softmax, evaluation function should be softmax_cross_entropy_loss'
        self.Wy = np.random.uniform(self.weights_left_limit, self.weights_right_limit, size=(self.layers[1][0], self.layers[0][0]))
        self.by = np.ones((self.layers[1][0],1)) * self.bias_init
        self.Wx = np.random.uniform(self.weights_left_limit, self.weights_right_limit, size=(self.layers[0][0], self.input_dim))
        self.Wh = np.random.uniform(self.weights_left_limit, self.weights_right_limit, size=(self.layers[0][0], self.layers[0][0]))
        self.bh = np.ones((self.layers[0][0],1)) * self.bias_init
        assert self.seq_len > 0, '[Error] sequence length should be greater than zero'

    def forward(self, X, trace=False):
        # X shape should be (t, (input_dim, n_samples))
        assert set(X.keys()) == set(range(self.seq_len)), '[Error] X should have {0} sequences'.format(self.seq_len)
        H, Y, derivatives = {}, {}, {}
        n_samples = X[0].shape[1]
        for t in range(self.seq_len):
            assert X[t].shape == (self.input_dim, n_samples), '[Error] X[{0}] shape is wrong'.format(t)
            H[t] = np.dot(self.Wx, X[t]) + np.dot(self.Wh, H[t-1] if t > 0 else np.zeros((self.layers[0][0], n_samples))) + self.bh
            H[t] = self.layers[0][1](H[t])
            assert H[t].shape == (self.layers[0][0], n_samples), '[Error] ht shape is wrong'
            Y[t] = np.dot(self.Wy, H[t]) + self.by
            Y[t] = self.layers[1][1](Y[t])
            assert Y[t].shape == (self.layers[1][0], n_samples), '[Error] Y[{0}] shape is wrong'.format(t)
            if trace:
                derivatives[t] = [self.layers[0][1](H[t], derivative=True), self.layers[1][1](Y[t], derivative=True)]
        if trace:
            return H, Y, derivatives
        else:
            return Y

    def backward(self, X, targets, verbose=False):
        assert set(X.keys()) == set(targets.keys()) == set(range(self.seq_len)), '[Error] X and targets should have same sequence'
        H, Y, derivatives = self.forward(X, trace=True)
        cost_derivative = self.cost_function(Y, targets, derivative=True)
        dL_dWx = np.zeros((self.layers[0][0], self.input_dim))
        dL_dWh = np.zeros((self.layers[0][0], self.layers[0][0]))
        dL_dWy = np.zeros((self.layers[1][0], self.layers[0][0]))
        dL_dby = np.zeros((self.layers[1][0], 1))
        dL_dbh = np.zeros((self.layers[0][0], 1))

        for t in reversed(range(self.seq_len)):
            assert X[t].shape == (self.input_dim, self.batch_size) and targets[t].shape == (self.layers[1][0], self.batch_size), \
                '[Error] X[{0}] shape or targets[{0}] shape is wrong'.format(t)
            # dLt_dZt = cost_derivative[t] * derivatives[t][-1]
            dLt_dZt = cost_derivative[t]
            assert dLt_dZt.shape == (self.layers[1][0], self.batch_size), '[Error] dLt_dal shape is wrong'
            tmp = np.dot(dLt_dZt, H[t].T) / float(self.batch_size)
            assert tmp.shape == (self.layers[1][0], self.layers[0][0]), '[Error] tmp shape is wrong'
            dL_dWy += tmp
            dL_dby += (np.sum(dLt_dZt, axis=1) / float(self.batch_size)).reshape((self.layers[1][0], 1))

            dLt_dWx = np.zeros((self.layers[0][0], self.input_dim))
            dLt_dWh = np.zeros((self.layers[0][0], self.layers[0][0]))
            dLt_dbh = np.zeros((self.layers[0][0], 1))
            for l in range(t, -1, -1):
                dLt_dal = np.dot(self.Wy.T, dLt_dZt) * derivatives[l][0]
                tmp1 = np.dot(dLt_dal, X[l].T) / float(self.batch_size)
                assert tmp1.shape == (self.layers[0][0], self.input_dim), '[Error] tmp1 shape is wrong'
                dLt_dWx += tmp1
                tmp2 = np.dot(dLt_dal, H[l-1].T if l > 0 else np.zeros((self.layers[0][0], self.batch_size)).T) / float(self.batch_size)
                assert tmp2.shape == (self.layers[0][0], self.layers[0][0]), '[Error] tmp2 shape is wrong'
                dLt_dWh += tmp2
                dLt_dbh += (np.sum(dLt_dal, axis=1) / float(self.batch_size)).reshape((self.layers[0][0], 1))

            dL_dWx += dLt_dWx
            dL_dWh += dLt_dWh
            dL_dbh += dLt_dbh

        if verbose:
            print('[Backpropagate] norm of Wx gradient: {0}, Wh gradient: {1}, Wy gradient: {2}'.format(np.linalg.norm(dL_dWx), np.linalg.norm(dL_dWh), np.linalg.norm(dL_dWy)))

        self.Wx -= self.learning_rate * dL_dWx
        self.Wh -= self.learning_rate * dL_dWh
        self.Wy -= self.learning_rate * dL_dWy
        self.bh -= self.learning_rate * dL_dbh
        self.by -= self.learning_rate * dL_dby


    def train(self, X_train, Y_train, X_validate, Y_validate, evaluate_function=softmax_cross_entropy_loss_allseq, n_epoch=50,
              verbose=False):
        if verbose:
            print('[INFO] Start training ... ')
        assert set(X_train.keys()) == set(X_validate.keys()) == set(Y_train.keys()) == set(Y_validate.keys()) == set(range(
            self.seq_len)), '[Error] inputs should have same length of sequence'
        n_samples = X_train[0].shape[1]
        self.train_error = np.zeros(n_epoch)
        self.validate_error = np.zeros(n_epoch)
        for e in range(n_epoch):
            start_ind = 0
            for i in range(n_samples / self.batch_size):
                X_train_batch, Y_train_batch = {}, {}
                for t in range(self.seq_len):
                    X_train_batch[t] = X_train[t][:, start_ind:start_ind+self.batch_size]
                    Y_train_batch[t] = Y_train[t][:, start_ind:start_ind+self.batch_size]
                    assert X_train_batch[t].shape == (self.input_dim, self.batch_size), '[Error] X_train_batch size is wrong'
                    assert Y_train_batch[t].shape == (self.layers[1][0], self.batch_size), '[Error] Y_train_batch size is wrong'
                self.backward(X_train_batch, Y_train_batch)
                start_ind += self.batch_size

            train_output = self.forward(X_train)
            validate_output = self.forward(X_validate)
            self.train_error[e] = evaluate_function(train_output, Y_train, derivative=False)
            self.validate_error[e] = evaluate_function(validate_output, Y_validate, derivative=False)
            if verbose:
                print('[Train Epoch {0}] train error is {1}, validate error is {2}'.format(e, self.train_error[e], self.validate_error[e]))

        return self.train_error, self.validate_error




