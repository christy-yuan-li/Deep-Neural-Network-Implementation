import numpy as np
from matplotlib import pyplot as plt
import pickle
from evaluation_functions import *


class DRBM(object):
    def __init__(self, input_dim, layers, learning_rate, persistent_chains, batch_size, weights_left_limit=-0.1, weights_right_limit=0.1,
                 bias_term_init=0.1, random_seed=1):
        assert len(layers) >= 1, '[Error] DRBM model must have larger than or equal to 2 layers'
        self.input_dim = input_dim
        self.n_layers = len(layers)
        self.layers = layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.weights_left_limit = weights_left_limit
        self.weights_right_limit = weights_right_limit
        self.bias_term_init = bias_term_init
        self.persistent_chains = persistent_chains
        self.init_weights_bias()
        self.init_persistent_CD()
        self.train_error = None
        self.validate_error = None

    def init_weights_bias(self):
        self.weights = []
        self.bias = []
        self.bias.append(np.ones((self.input_dim, 1)) * self.bias_term_init)
        for i, h_dim in enumerate(self.layers):
            layer_bias = np.ones((self.layers[i], 1)) * self.bias_term_init
            self.bias.append(layer_bias)
            if i == 0:
                layer_weight = np.random.uniform(self.weights_left_limit, self.weights_right_limit, size=(self.input_dim, self.layers[i]))
                assert layer_weight.shape == (self.input_dim, self.layers[i]), '[Error] layer_weight shape is wrong'
                self.weights.append(layer_weight)
            else:
                layer_weight = np.random.uniform(self.weights_left_limit, self.weights_right_limit, size=(self.layers[i-1], self.layers[i]))
                assert layer_weight.shape == (self.layers[i-1], self.layers[i]), '[Error] layer_weight shape is wrong'
                self.weights.append(layer_weight)

    def init_persistent_CD(self):
        self.persistent_v = {}
        self.persistent_hs = {}
        for k in range(self.persistent_chains):
            self.persistent_v[k] = np.random.rand(self.input_dim, 1)
            self.persistent_v[k] = self.binary_sample(self.persistent_v[k])
        for k in range(self.persistent_chains):
            persistent_h = []
            for i, h_dim in enumerate(self.layers):
                persistent_h.append(np.random.rand(h_dim, 1))
                assert persistent_h[i].shape == (h_dim, 1), '[Error] persistent_h shape is wrong'
                persistent_h[i] = self.binary_sample(persistent_h[i])
            self.persistent_hs[k] = persistent_h

    def get_prob(self, layer_index, prev_layer=None, post_layer=None):
        assert (layer_index == -1 and prev_layer is None and post_layer is not None) or (-1 < layer_index < self.n_layers-1 and prev_layer is not None and post_layer is not None) or (layer_index == self.n_layers-1 and prev_layer is not None and post_layer is None), '[Error] prev_layer or post_layer error'
        n_samples = prev_layer.shape[1] if prev_layer is not None else post_layer.shape[1]
        E = self.bias[layer_index + 1]
        if prev_layer is not None:
            assert prev_layer.shape[0] == (self.layers[layer_index-1] if layer_index > 0 else self.input_dim), '[Error] prev_layer shape is wrong'
            E = E + np.dot(prev_layer.T, self.weights[layer_index]).T
        if post_layer is not None:
            assert post_layer.shape[0] == self.layers[layer_index+1], '[Error] post_layer shape is wrong'
            E = E + np.dot(self.weights[layer_index+1], post_layer)
        ex = np.exp(E)
        assert ex.shape == (self.layers[layer_index] if layer_index >= 0 else self.input_dim, n_samples), '[Error] ex shape is wrong'
        p = np.divide(ex, 1.0+ex)
        return p


    def mean_filed(self, v, n_steps=10, verbose=True):
        assert v.shape[0] == self.input_dim, '[Error] v shape is wrong'
        n_samples = v.shape[1]
        mu, prev_mu = [None] * self.n_layers, [None] * self.n_layers
        for i, h_dim in enumerate(self.layers):
            mu[i] = np.random.rand(h_dim, n_samples)
        for l in range(n_steps):
            for i, h_dim in enumerate(self.layers):
                prev_mu[i] = mu[i]
                if i == 0:
                    mu[i] = self.get_prob(i, prev_layer=v, post_layer=mu[i+1])
                elif i < self.n_layers-1:
                    mu[i] = self.get_prob(i, prev_layer=mu[i-1], post_layer=mu[i+1])
                else:
                    mu[i] = self.get_prob(i, prev_layer=mu[i-1])
            if verbose:
                print('[Mean Filed] Step {0}:  mu[0] changed {1}, mu[1] changed {2}'.format(l, np.linalg.norm(prev_mu[0] - mu[0]),
                                                                                           np.linalg.norm(prev_mu[1] - mu[1])))
        if verbose:
            print('[INFO] finished mean filed')
        return mu

    def forward(self, X, last_layer_prob=False):
        assert X.shape[0] == self.input_dim, '[Error] X shape is wrong'
        mu = self.mean_filed(X, verbose=False)
        return self.forward_helper(X, mu, last_layer_prob=last_layer_prob)

    def forward_helper(self, v_tilta, h_tilta, last_layer_prob=False):
        n_samples = v_tilta.shape[1]
        for i, h_dim in enumerate(self.layers):
            prob_h_layer = self.get_prob(i, prev_layer=(v_tilta if i == 0 else h_tilta[i-1]), post_layer=(h_tilta[i+1] if i <
                                                                                                    self.n_layers-1 else None) )
            h_tilta[i] = self.sample(prob_h_layer)
            assert h_tilta[i].shape == (self.layers[i], n_samples), '[Error] h_tilta[{0}] shape is wrong'.format(i)
        prob_v_tilta = self.get_prob(-1, prev_layer=None, post_layer=h_tilta[0])
        v_tilta = self.sample(prob_v_tilta)
        assert v_tilta.shape == (self.input_dim, n_samples), '[Error] v_tilta shape is wrong'

        if last_layer_prob:
            return prob_v_tilta
        else:
            return v_tilta, h_tilta

    def persistent_CD(self, n_samples, verbose=True):
        v_tilta = np.zeros((n_samples, self.input_dim))
        h_tilta = []
        for i, h_dim in enumerate(self.layers):
            h_tilta.append(np.zeros((n_samples, h_dim)))

        for s in range(n_samples):
            for k in range(self.persistent_chains):
                self.persistent_v[k], self.persistent_hs[k] = self.forward_helper(self.persistent_v[k], self.persistent_hs[k])
                v_tilta[s, :] = (v_tilta[s, :].reshape((self.input_dim, 1)) + self.persistent_v[k]).reshape((1, self.input_dim))
                for i, h_dim in enumerate(self.layers):
                    h_tilta[i][s, :] = (h_tilta[i][s,:].reshape((h_dim, 1)) + self.persistent_hs[k][i]).reshape((1, h_dim))

        v_tilta /= float(self.persistent_chains)
        for i in range(self.n_layers):
            h_tilta[i] /= float(self.persistent_chains)
            h_tilta[i] = h_tilta[i].T
        if verbose:
            print('[INFO] finished Gibbs Sampling')
        return v_tilta.T, h_tilta

    def parameter_update(self, v, mu, v_tilta, h_tilta, verbose=True):
        n_samples = v.shape[1]
        assert v_tilta.shape == v.shape, '[Error] v_tilta shape should be the same as v shape'
        for i in range(self.n_layers):
            assert h_tilta[i].shape[0] == mu[i].shape[0] == self.layers[i], '[Error] h_tilta, mu shape[0] should be equal to ' \
                                                                            'corresponding layer dimension'
            assert h_tilta[i].shape[1] == mu[i].shape[1] == n_samples, '[Error] number of sampels dose not match with given variable'

        delta_bias_v = (np.sum(v - v_tilta, axis=1)/float(n_samples)).reshape((self.input_dim, 1))
        self.bias[0] = self.bias[0] + self.learning_rate * delta_bias_v
        for i, h_dim in enumerate(self.layers):
            if i == 0:
                delta_W_layer = np.dot(v, mu[i].T)/float(n_samples) - np.dot(v_tilta, h_tilta[i].T)/float(n_samples)
            else:
                delta_W_layer = np.dot(mu[i-1], mu[i].T)/float(n_samples) - np.dot(h_tilta[i-1], h_tilta[i].T)/float(n_samples)
            assert delta_W_layer.shape == self.weights[i].shape, '[Error] delta_W_shape is wrong'
            self.weights[i] = self.weights[i] + self.learning_rate * delta_W_layer
            delta_bias_layer = (np.sum(mu[i] - h_tilta[i], axis=1)/float(n_samples)).reshape((h_dim, 1))
            assert delta_bias_layer.shape == self.bias[i+1].shape, '[Error] delta_bias_layer shape is wrong'
            self.bias[i+1] = self.bias[i+1] + self.learning_rate * delta_bias_layer

        if verbose:
            print('[INFO] gradient norm, delta_W of last layer: {0}, delta_bias of v: {1}, delta_bias of last layer: {2}'.format(
                np.linalg.norm(delta_W_layer), np.linalg.norm(delta_bias_v), np.linalg.norm(delta_bias_layer)))

    def compute_cross_entropy_error(self, X, prob_v):
        assert X.shape[0] == self.input_dim and X.shape == prob_v.shape, '[Error] X shape is wrong'
        n_samples = X.shape[1]
        error = 0.0
        for i in range(n_samples):
            error += -1.0 * (np.dot(X[:, i].T, np.log(prob_v[:,i])) + np.dot((1.0 - X[:, i]).T, np.log(1.0 - prob_v[:,i])))
        return error / float(n_samples)

    def update(self, X_batch):
        assert X_batch.shape == (self.batch_size, self.input_dim), '[Error] X_batch shape is wrong'
        mu = self.mean_filed(X_batch.T, verbose=False)
        v_tilta, h_tilta = self.persistent_CD(X_batch.shape[0], verbose=False)
        self.parameter_update(X_batch.T, mu, v_tilta, h_tilta, verbose=False)

    def train(self, X_train, X_validate, n_epoch=50, verbose=True):
        assert X_train.shape[1] == self.input_dim, '[Error] X_train dimension is wrong'
        assert X_validate.shape[1] == self.input_dim, '[Error] X_validate shape is wrong'
        self.train_error = np.zeros((n_epoch, 1))
        self.validate_error = np.zeros((n_epoch, 1))
        if verbose:
            print('[Info] Start training ...')
        for e in range(n_epoch):
            for k in range(0, int(len(X_train) / self.batch_size)):
                self.update(X_train[k * self.batch_size:((k + 1) * self.batch_size)])

            prob_v_train = self.forward(X_train.T, last_layer_prob=True)
            prob_v_validate = self.forward(X_validate.T, last_layer_prob=True)
            self.train_error[e] = self.compute_cross_entropy_error(X_train.T, prob_v_train)
            self.validate_error[e] = self.compute_cross_entropy_error(X_validate.T, prob_v_validate)
            if verbose:
                print('\n[Train] epoch {0}, train error is {1}, validate error is {2}\n'.format(e, self.train_error[e], self.validate_error[e]))

        return self.train_error, self.validate_error

    def save_model(self, name):
        pickle.dump(self, open('{0}.pkl'.format(name), 'wb'))

    def load_model(self, name):
        model = pickle.load(open(name, 'rb'))
        return model

    def plot_W(self, layer_index=0, figname='plots/DRBM_w'):
        f, axarr = plt.subplots(10, 10, sharex='col', sharey='row')
        for i in range(self.layers[layer_index]):
            r = i / 10
            c = i % 10
            w1 = self.weights[layer_index][:,i].reshape(28, 28)
            axarr[r, c].imshow(w1, cmap='gray')
        plt.show()
        f.savefig('{0}{1}.pdf'.format(figname, layer_index+1))

    def sample(self, prob_x):
        # x_sample =np.array([[1.0 if prob_x[i,j] >= 0.5 else 0.0 for j in range(len(prob_x[0]))] for i in range(len(
        #     prob_x))])
        # assert x_sample.shape == prob_x.shape, '[Error] x_sample shape is wrong'
        return prob_x

    def binary_sample(self, prob_x):
        x_sample = np.array([[1.0 if prob_x[i, j] >= 0.5 else 0.0 for j in range(len(prob_x[0]))] for i in range(len(
            prob_x))])
        assert x_sample.shape == prob_x.shape, '[Error] x_sample shape is wrong'
        return x_sample

