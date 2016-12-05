import numpy as np
from matplotlib import pyplot as plt
import pickle

class RBM(object):
    def __init__(self, input_dim, hidden_dim, learning_rate=0.02, Gibbs_interations=5, batch_size=1, weights_left_limit=-0.1,
                 weights_right_limit=0.1, bias_term_init=0.1, persistent_CD_chains=10, CD_method='persistent_CD'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.Gibbs_interations = Gibbs_interations
        self.batch_size = batch_size
        np.random.seed(1)
        self.weights_left_limit = weights_left_limit
        self.weights_right_limit = weights_right_limit
        self.W = np.random.uniform(self.weights_left_limit, self.weights_right_limit, size=(self.hidden_dim, self.input_dim))
        self.c = np.ones((self.input_dim, 1)) * bias_term_init
        self.b = np.ones((self.hidden_dim, 1)) * bias_term_init
        self.train_error = None
        self.validate_error = None
        self.CD_method = CD_method
        self.persistent_CD_v = {}
        self.persistent_CD_h = {}
        self.persistent_chains = persistent_CD_chains
        for k in range(self.persistent_chains):
            self.persistent_CD_v[k] = np.random.rand(self.input_dim, 1)
            self.persistent_CD_h[k] = np.random.rand(self.hidden_dim, 1)
            self.persistent_CD_v[k] = self.binary_sample(self.persistent_CD_v[k])
            self.persistent_CD_h[k] = self.binary_sample(self.persistent_CD_h[k])

    def get_first_layer_weights(self):
        W = np.hstack((self.W, self.b))
        weight_array = np.array(W.flat)
        return weight_array

    def get_prob_H_X(self, X):
        assert X.shape[1] == self.input_dim, '[Error] x shape is wrong'
        e = np.exp(-np.dot(self.W, X.T) - self.b)
        assert e.shape[0] == self.hidden_dim, '[Error] e shape is wrong'
        prob_h_x = np.divide(1.0, 1.0+e)
        assert prob_h_x.shape[0] == self.hidden_dim, '[Error] prob_h_x shape is wrong'
        return prob_h_x

    def get_prob_X_H(self, H):
        assert H.shape[0] == self.hidden_dim, '[Error] h shape wrong'
        e = np.exp(-self.c.T - np.dot(H.T, self.W))
        prob_x_h = np.divide(1.0, 1.0+e)
        assert prob_x_h.shape[1] == self.input_dim, '[Error] prob_x_h shape is wrong'
        return prob_x_h

    def update(self, X):
        gradient_W, gradient_b, gradient_c = self.compute_gradient(X)
        assert gradient_W.shape == self.W.shape, '[Error] gradient_W shape is different from self.W shape'
        self.W -= self.learning_rate * gradient_W
        self.b -= self.learning_rate * gradient_b
        self.c -= self.learning_rate * gradient_c

    def compute_gradient(self, X):
        assert X.shape == (self.batch_size, self.input_dim), '[Error] x shape is wrong'
        X = np.array(X)
        prob_h_x = self.get_prob_H_X(X)
        if self.CD_method == 'persistent_CD':
            X_tilta = self.persistent_CD(X.shape[0], verbose=False)
        elif self.CD_method == 'contrastive_divergence':
            X_tilta = self.contrastive_divergence(X, self.Gibbs_interations)

        prob_h_x_tilta = self.get_prob_H_X(X_tilta)
        gradient_W = (-np.dot(prob_h_x, X) + np.dot(prob_h_x_tilta, X_tilta)) / float(self.batch_size)
        assert gradient_W.shape == (self.hidden_dim, self.input_dim), '[Error] gradient_W shape is wrong'
        gradient_b = np.sum(-prob_h_x + prob_h_x_tilta, axis=1).reshape(self.hidden_dim,1) / float(self.batch_size)
        assert gradient_b.shape == (self.hidden_dim, 1), '[Error] gradient_b shape is wrong'
        gradient_c = np.sum(-X + X_tilta, axis=0).reshape(self.input_dim, 1) / float(self.batch_size)
        assert gradient_c.shape == (self.input_dim, 1), '[Error] gradient_c shape is wrong'

        return gradient_W, gradient_b, gradient_c

    def binary_sample(self, prob_x):
        x_sample = np.array([[1.0 if prob_x[i, j] > 0.5 else 0.0 for j in range(len(prob_x[0]))] for i in range(len(
            prob_x))])
        assert x_sample.shape == prob_x.shape, '[Error] x_sample shape is wrong'
        return x_sample

    def persistent_CD(self, n_samples, verbose=True):
        v_tilta = np.zeros((n_samples, self.input_dim))
        h_tilta = np.zeros((n_samples, self.hidden_dim))

        for i in range(n_samples):
            for k in range(self.persistent_chains):
                self.persistent_CD_h[k] = self.get_prob_H_X(self.persistent_CD_v[k].T)
                # self.persistent_CD_h[k] = self.binary_sample(self.persistent_CD_h[k])
                self.persistent_CD_v[k] = self.get_prob_X_H(self.persistent_CD_h[k]).T
                # self.persistent_CD_v[k] = self.binary_sample(self.persistent_CD_v[k])

                v_tilta[i, :] = (v_tilta[i, :].reshape((self.input_dim, 1)) + self.persistent_CD_v[k]).reshape((1, self.input_dim))
                h_tilta[i, :] = (h_tilta[i, :].reshape((self.hidden_dim, 1)) + self.persistent_CD_h[k]).reshape((1, self.hidden_dim))

        v_tilta /= float(self.persistent_chains)
        h_tilta /= float(self.persistent_chains)

        if verbose:
            print('[INFO] finished persistent contrastive divergence')
        return v_tilta


    def contrastive_divergence(self, X, K):
        assert X.shape[1] == self.input_dim, '[Error] x shape is wrong'
        n_samples = X.shape[0]
        X_tilta = X
        for k in range(K):
            prob_h_tilta = self.get_prob_H_X(X_tilta)
            h_tilta = np.zeros((self.hidden_dim, n_samples))
            for j in range(self.hidden_dim):
                for i in range(n_samples):
                    h_tilta[j, i] = 1.0 if prob_h_tilta[j,i] > 0.5 else 0.0

            prob_x_tilta = self.get_prob_X_H(h_tilta)
            X_tilta = np.zeros((n_samples, self.input_dim))
            for i in range(n_samples):
                for j in range(self.input_dim):
                    X_tilta[i,j] = 1.0 if prob_x_tilta[i,j] > 0.5 else 0.0
        return X_tilta

    def compute_cross_entropy_error(self, X):
        assert X.shape[1] == self.input_dim, '[Error] x shape is wrong'
        n_samples = X.shape[0]
        X = X.T
        h_real = np.dot(self.W, X)
        h = np.zeros((self.hidden_dim, n_samples))
        for j in range(self.hidden_dim):
            for i in range(n_samples):
                h[j, i] = 1.0 if h_real[j, i] > 0.5 else 0.0
        prob_x_h = self.get_prob_X_H(h)
        error = 0.0
        for i in range(n_samples):
            error += -1.0 * (np.dot(X[:,i].T, np.log(prob_x_h[i].T)) + np.dot((1.0-X[:,i]).T, np.log(1.0 - prob_x_h[i].T)))
        return error/float(n_samples)

    def plot_W(self):
        f, axarr = plt.subplots(10, 10, sharex='col', sharey='row')
        for i in range(self.hidden_dim):
            r = i/10
            c = i%10
            w = self.W[i].reshape(28, 28)
            axarr[r,c].imshow(w, cmap='gray')
        plt.show()
        f.savefig('./w.pdf')

    def sample_Gibbs(self, a, n_chains=100, n_steps=1000):
        v_tilta = np.zeros((n_chains, self.input_dim))
        persistent_CD_v = {}
        persistent_CD_h = {}

        for k in range(n_chains):
            persistent_CD_v[k] = a[k].reshape((self.input_dim, 1))

            for i in range(n_steps):
                persistent_CD_h[k] = self.get_prob_H_X(persistent_CD_v[k].T)
                # persistent_CD_h[k] = self.binary_sample(persistent_CD_h[k])
                persistent_CD_v[k] = self.get_prob_X_H(persistent_CD_h[k]).T
                # persistent_CD_v[k] = self.binary_sample(persistent_CD_v[k])

            v_tilta[k, :] = persistent_CD_v[k].T

        X_tilta = v_tilta
        assert X_tilta.shape == (n_chains, self.input_dim), '[Error] X_tilta shape is wrong'

        f, axarr = plt.subplots(10, 10, sharex='col', sharey='row')
        for k in range(n_chains):
            r = k / 10
            c = k % 10
            im = persistent_CD_v[k].reshape(28, 28)
            axarr[r, c].imshow(im, cmap='gray')
        plt.show()
        f.savefig('./RBM_Gibbs_samples.pdf')

    def train(self, X_train, X_validate, n_epoch=50):
        self.train_error = np.zeros((n_epoch, 1))
        self.validate_error = np.zeros((n_epoch, 1))
        print('[Info] Start training ...')
        for e in range(n_epoch):
            for k in range(0, int(len(X_train) / self.batch_size)):
                self.update(X_train[k * self.batch_size:((k + 1) * self.batch_size)])
            self.train_error[e] = self.compute_cross_entropy_error(X_train)
            self.validate_error[e] = self.compute_cross_entropy_error(X_validate)
            print('[Train] epoch {0}, train error is {1}, validate error is {2}'.format(e, self.train_error[e], self.validate_error[e]))
        return self.train_error, self.validate_error

    def save_model(self, name):
        pickle.dump(self, open('{0}.pkl'.format(name), 'wb'))

    def load_model(self, name):
        model = pickle.load(open(name, 'rb'))
        return model