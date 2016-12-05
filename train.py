import numpy as np
CLASS_NUM = 10
from RBM import *
from RNN import *
from NN import *
from Autoencoder import *
from matplotlib import pyplot as plt
from DRBM import *
import pickle
from utils import *


if __name__ == "__main__":

    ############################### input MNIST #########################
    # X, Y = input_data(CLASS_NUM)
    # print('[INFO] training samples: {0}, validation samples: {1}'.format(len(X['train']), len(X['valid'])))
    #
    # a = []
    # for i in range(100):
    #     a.append(X['test'][25*i,:])
    # a = np.array(a[::-1])

    ############################## input tiny-shakespeare #########################
    seq_len = 10
    X, Y, samples, input_dim = input_shakespeare(seq_len=seq_len, verbose=False)

    ################################ RNN #################################
    model = RNN({'input_dim':input_dim, 'layers':[(100, sigmoid), (input_dim, sigmoid)], 'seq_len':seq_len, 'batch_size':10,
                'learning_rate':0.1, 'random_seed':100, 'cost_function':sigmoid_cross_entropy_loss_allseq})
    model.train(X['train'], Y['train'], X['val'], Y['val'], n_epoch=50, verbose=True, evaluate_function=sigmoid_cross_entropy_loss_allseq)


    ################################ DRBM ################################

    # model = DRBM(784, [100, 100], learning_rate=0.1, persistent_chains=5, batch_size=10, random_seed=100)
    # model.train(X['train'], X['valid'], n_epoch=100)
    # model.save_model('models/DRBM_100_100')
    # model = pickle.load(open('models/DRBM_400_400.pkl', 'rb'))
    # model.train_error[5] = model.train_error[4]
    # model.validate_error[5] = model.validate_error[4]
    # plot_error(model.train_error, model.validate_error, 'plots/DRBM_error_400_400')
    # model.plot_W(layer_index=0, figname='plots/DRBM_w')
    #
    # np.random.seed(100)
    # model = pickle.load(open('models/DRBM.pkl', 'rb'))
    # model.sample_Gibbs(a)


    ################################  RBM  ###############################

    # model = RBM(784, 100, learning_rate=0.1, Gibbs_interations=5, batch_size=30)
    # train_error, validate_error = model.train(X['train'], X['valid'], n_epoch=100)
    # plot_error(train_error, validate_error, 'plots/RBM_error_curves_100_20')
    # model = pickle.load(open('./models/RBM_model.pkl', 'rb'))
    # # np.random.seed(54)
    #
    # model.sample_Gibbs(np.array(a[::-1]))
    # model.save_model('models/RBM_model')

    ###############################  Autoencoder  ###############################

    # model = Autoencoder({'input_dim': 784, 'layers': [(100, sigmoid), (784, sigmoid)], 'random_seed': 1, 'learning_rate': 0.1, 'batch_size': 30})
    # train_error, validate_error = model.train(X['train'], X['train'], X['valid'], X['valid'], n_epoch=30)
    # model.plot_W('plots/autoencoder_W')
    # plot_error(train_error, validate_error, 'plots/autoencoder_error_curve_500')
    # model.save_model('models/autoencoder_model')


    ##############################  Denoising Autoencoder  ##########################
    # Y_train = X['train']
    # Y_validate = X['valid']
    # X_train = add_noise(X['train'])
    # X_validate = add_noise(X['valid'])
    # model = Autoencoder({'input_dim': 784, 'layers': [(500, sigmoid), (784, sigmoid)], 'random_seed': 1, 'learning_rate': 0.1, 'batch_size': 30})
    # train_error, validate_error = model.train(X_train, Y_train, X_validate, Y_validate, n_epoch=50)
    # model.plot_W('plots/denoising_autoencoder_W')
    # plot_error(train_error, validate_error, 'plots/denoiseing_autoencoder_error_curve_500')
    # model.save_model('models/denoising_autoencoder_model')


    ################################ Neural Network ##############################
    # model = NN({'input_dim': 784, 'layers': [(100, sigmoid), (10, sigmoid)], 'random_seed': 1, 'learning_rate': 0.1, 'batch_size': 30})
    # # model.set_first_layer_weights(autoencoder.get_first_layer_weights())
    # train_error, validate_error = model.train(X['train'], Y['train'], X['valid'], Y['valid'], n_epoch=100,
    #                                     cost_function=sigmoid_cross_entropy_loss, evaluate_function=classification_accuracy)
    # plot_error(train_error, validate_error, 'plots/neural_network_error_denoiseing_autoencoder_init_acc.pdf')
    # model.save_model('models/NN_model')
