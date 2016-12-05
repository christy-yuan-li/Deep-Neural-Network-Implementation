import numpy as np
from matplotlib import pyplot as plt
import pickle
from evaluation_functions import *


class DRBM(object):
    def __init__(self, input_dim, h1_dim, h2_dim, learning_rate, persistent_chains, batch_size, weights_left_limit=-0.1, weights_right_limit=0.1, bias_term_init=0.1, random_seed=1):
        self.input_dim = input_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.b = np.ones((self.input_dim, 1)) * bias_term_init
        self.c = np.ones((self.h1_dim, 1)) * bias_term_init
        self.d = np.ones((self.h2_dim, 1)) * bias_term_init
        self.weights_left_limit = weights_left_limit
        self.weights_right_limit = weights_right_limit
        self.W1 = np.random.uniform(self.weights_left_limit, self.weights_right_limit, size=(self.input_dim, self.h1_dim))
        self.W2 = np.random.uniform(self.weights_left_limit, self.weights_right_limit, size=(self.h1_dim, self.h2_dim))
        self.train_error = None
        self.validate_error = None
        self.persistent_CD_v = {}
        self.persistent_CD_h1 = {}
        self.persistent_CD_h2 = {}
        self.persistent_chains = persistent_chains
        for k in range(self.persistent_chains):
            self.persistent_CD_v[k] = np.random.rand(self.input_dim, 1)
            self.persistent_CD_h1[k] = np.random.rand(self.h1_dim, 1)
            self.persistent_CD_h2[k] = np.random.rand(self.h2_dim, 1)
            self.persistent_CD_v[k] = self.binary_sample(self.persistent_CD_v[k])
            self.persistent_CD_h1[k] = self.binary_sample(self.persistent_CD_h1[k])
            self.persistent_CD_h2[k] = self.binary_sample(self.persistent_CD_h2[k])


    def get_P_v_h1(self, h1):
        n_samples = h1.shape[1]
        assert h1.shape[0] == self.h1_dim, '[Error] h1 shape is wrong'
        e = np.exp(np.dot(self.W1, h1) + self.b)
        assert e.shape == (self.input_dim, n_samples), '[Error] e shape is wrong'
        p = np.divide(e, 1.0+e)
        assert p.shape == (self.input_dim, n_samples), '[Error] p shape is wrong'
        return p

    def get_P_h2_h1(self, h1):
        n_samples = h1.shape[1]
        assert h1.shape[0] == self.h1_dim, '[Error] h1 shape is wrong'
        e = np.exp(np.dot(h1.T, self.W2).T + self.d)
        assert e.shape == (self.h2_dim, n_samples), '[Error] e shape is wrong'
        p = np.divide(e, 1.0+e)
        assert p.shape == (self.h2_dim, n_samples), '[Error] p shape is wrong'
        return p

    def get_P_h1_vh2(self, v, h2):
        n_samples = v.shape[1]
        assert h2.shape[0] == self.h2_dim, '[Error] h2 shape is wrong'
        assert v.shape[0] == self.input_dim, '[Erorr] v shape is wrong'
        e = np.exp(np.dot(v.T, self.W1).T + np.dot(self.W2, h2) + self.c)
        assert e.shape == (self.h1_dim, n_samples), '[Error] e shape is wrong'
        p = np.divide(e, 1.0+e)
        assert p.shape == (self.h1_dim, n_samples), '[Error] p shape is wrong'
        return p

    def mean_filed(self, v, n_steps=10, verbose=True):
        assert v.shape[0] == self.input_dim, '[Error] v shape is wrong'
        n_samples = v.shape[1]
        mu1 = np.random.uniform(self.weights_left_limit, self.weights_right_limit, size=(self.h1_dim, n_samples))
        mu2 = np.random.uniform(self.weights_left_limit, self.weights_right_limit, size=(self.h2_dim, n_samples))
        for l in range(n_steps):
            prev_mu1 = mu1
            prev_mu2 = mu2
            mu1 = self.get_P_h1_vh2(v, mu2)
            mu2 = self.get_P_h2_h1(mu1)
            if verbose:
                print('[Mean Filed] Step {0}:  mu1 changed {1}, m2 changed {2}'.format(l, np.linalg.norm(prev_mu1 - mu1), np.linalg.norm(prev_mu2 - mu2)))
        if verbose:
            print('[INFO] finished mean filed')
        return mu1, mu2

    def sample(self, prob_x):
        # x_sample =np.array([[1.0 if prob_x[i,j] >= 0.5 else 0.0 for j in range(len(prob_x[0]))] for i in range(len(
        #     prob_x))])
        # assert x_sample.shape == prob_x.shape, '[Error] x_sample shape is wrong'
        return prob_x

    def binary_sample(self, prob_x):
        x_sample = np.array([[1.0 if prob_x[i,j] >= 0.5 else 0.0 for j in range(len(prob_x[0]))] for i in range(len(
            prob_x))])
        assert x_sample.shape == prob_x.shape, '[Error] x_sample shape is wrong'
        return x_sample

    def forward(self, X, last_layer_prob=False):
        assert X.shape[0] == self.input_dim, '[Error] X shape is wrong'
        mu1, mu2 = self.mean_filed(X, verbose=False)
        return self.forward_helper(X, mu2, last_layer_prob=last_layer_prob)


    def forward_helper(self, v_tilta, h2_tilta, last_layer_prob=False):
        n_samples = v_tilta.shape[1]
        prob_h1_tilta = self.get_P_h1_vh2(v_tilta, h2_tilta)
        h1_tilta = self.sample(prob_h1_tilta)
        prob_h2_tilta = self.get_P_h2_h1(h1_tilta)
        h2_tilta = self.sample(prob_h2_tilta)
        prob_v_tilta = self.get_P_v_h1(h1_tilta)
        v_tilta = self.sample(prob_v_tilta)
        assert v_tilta.shape == (self.input_dim, n_samples), '[Error] v_tilta shape is wrong'
        assert h1_tilta.shape == (self.h1_dim, n_samples), '[Error] h1_tilta shape is wrong'
        assert h2_tilta.shape == (self.h2_dim, n_samples), '[Error] h2_tilta shape is wrong'
        if last_layer_prob:
            return prob_v_tilta
        else:
            return v_tilta, h1_tilta, h2_tilta

    def persistent_CD(self, n_samples, verbose=True):
        v_tilta = np.zeros((n_samples, self.input_dim))
        h1_tilta = np.zeros((n_samples, self.h1_dim))
        h2_tilta = np.zeros((n_samples, self.h2_dim))

        for i in range(n_samples):
            for k in range(self.persistent_chains):
                self.persistent_CD_v[k], self.persistent_CD_h1[k], self.persistent_CD_h2[k] = self.forward_helper(self.persistent_CD_v[k], self.persistent_CD_h2[k])
                v_tilta[i, :] = (v_tilta[i, :].reshape((self.input_dim, 1)) + self.persistent_CD_v[k]).reshape((1, self.input_dim))
                h1_tilta[i, :] = (h1_tilta[i, :].reshape((self.h1_dim, 1)) + self.persistent_CD_h1[k]).reshape((1, self.h1_dim))
                h2_tilta[i, :] = (h2_tilta[i, :].reshape((self.h2_dim, 1)) + self.persistent_CD_h2[k]).reshape((1, self.h2_dim))

        v_tilta /= float(self.persistent_chains)
        h1_tilta /= float(self.persistent_chains)
        h2_tilta /= float(self.persistent_chains)

        if verbose:
            print('[INFO] finished Gibbs Sampling')
        return v_tilta.T, h1_tilta.T, h2_tilta.T

    def parameter_update(self, v, mu1, mu2, v_tilta, h1_tilta, h2_tilta, verbose=True):
        assert v_tilta.shape[0] == v.shape[0] == self.input_dim and h1_tilta.shape[0] == mu1.shape[0] == self.h1_dim and h2_tilta.shape[0] == mu2.shape[0] == self.h2_dim, '[Error] input variables dimension do not match with setting'
        assert v.shape[1] == mu1.shape[1] == mu2.shape[1] == v_tilta.shape[1] == h1_tilta.shape[1] == h2_tilta.shape[1], '[Error] number of sampels do not match in given variables'
        n_samples = v.shape[1]
        delta_W1 = np.dot(v, mu1.T)/float(n_samples) - np.dot(v_tilta, h1_tilta.T)/float(n_samples)
        delta_W2 = np.dot(mu1, mu2.T)/float(n_samples) - np.dot(h1_tilta, h2_tilta.T)/float(n_samples)
        assert delta_W1.shape == self.W1.shape, '[Error] delta_W1_shape is wrong'
        assert delta_W2.shape == self.W2.shape, '[Error] delta_W2_shape is wrong'
        self.W1 = self.W1 + self.learning_rate * delta_W1
        self.W2 = self.W2 + self.learning_rate * delta_W2
        delta_b = (np.sum(v - v_tilta, axis=1)/float(n_samples)).reshape((self.input_dim, 1))
        delta_c = (np.sum(mu1 - h1_tilta, axis=1)/float(n_samples)).reshape((self.h1_dim, 1))
        delta_d = (np.sum(mu2 - h2_tilta, axis=1)/float(n_samples)).reshape((self.h2_dim, 1))
        assert delta_b.shape == self.b.shape, '[Error] delta_b shape is wrong'
        assert delta_c.shape == self.c.shape, '[Error] delta_c shape is wrong'
        assert delta_d.shape == self.d.shape, '[Error] delta_d shape is wrong'
        self.b = self.b + self.learning_rate * delta_b
        self.c = self.c + self.learning_rate * delta_c
        self.d = self.d + self.learning_rate * delta_d

        if verbose:
            print('[INFO] gradient norm, delta_W1: {0}, delta_W2: {1}, delta_b: {2}, delta_c: {3}, delta_d: {4}'.format(np.linalg.norm(
                delta_W1), np.linalg.norm(delta_W2), np.linalg.norm(delta_b), np.linalg.norm(delta_c), np.linalg.norm(delta_d)))

    def compute_cross_entropy_error(self, X, prob_v):
        assert X.shape[0] == self.input_dim and X.shape == prob_v.shape, '[Error] X shape is wrong'
        n_samples = X.shape[1]
        error = 0.0
        for i in range(n_samples):
            error += -1.0 * (np.dot(X[:, i].T, np.log(prob_v[:,i])) + np.dot((1.0 - X[:, i]).T, np.log(1.0 - prob_v[:,i])))
        return error / float(n_samples)

    def update(self, X_batch):
        assert X_batch.shape == (self.batch_size, self.input_dim), '[Error] X_batch shape is wrong'
        mu1, mu2 = self.mean_filed(X_batch.T, verbose=False)
        v_tilta, h1_tilta, h2_tilta = self.persistent_CD(X_batch.shape[0], verbose=False)
        self.parameter_update(X_batch.T, mu1, mu2, v_tilta, h1_tilta, h2_tilta, verbose=False)

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

    def plot_W1(self):
        f, axarr = plt.subplots(10, 10, sharex='col', sharey='row')
        for i in range(self.h1_dim):
            r = i / 10
            c = i % 10
            w1 = self.W1[:,i].reshape(28, 28)
            axarr[r, c].imshow(w1, cmap='gray')
        plt.show()
        f.savefig('./DRBM_w1.pdf')

    def sample_Gibbs(self, n_chains=100, steps=10):
        persistent_CD_v = {}
        persistent_CD_h1 = {}
        persistent_CD_h2 = {}
        for k in range(n_chains):
            persistent_CD_v[k] = np.random.rand(self.input_dim,1)
            persistent_CD_h1[k] = np.random.rand(self.h1_dim,1)
            persistent_CD_h2[k] = np.random.rand(self.h2_dim,1)
            persistent_CD_v[k] = self.binary_sample(persistent_CD_v[k])
            persistent_CD_h1[k] = self.binary_sample(persistent_CD_h1[k])
            persistent_CD_h2[k] = self.binary_sample(persistent_CD_h2[k])

            for i in range(steps):
                persistent_CD_v[k], persistent_CD_h1[k], persistent_CD_h2[k] = self.forward_helper(persistent_CD_v[k], persistent_CD_h2[k])
            persistent_CD_v[k] = self.binary_sample(persistent_CD_v[k])

        f, axarr = plt.subplots(10, 10, sharex='col', sharey='row')
        for k in range(n_chains):
            r = k / 10
            c = k % 10
            im = persistent_CD_v[k].reshape(28, 28)
            axarr[r, c].imshow(im, cmap='gray')
        plt.show()
        f.savefig('./DRBM_Gibbs_samples.pdf')










