from matplotlib import pyplot as plt
import numpy as np
import h5py

def input_data(CLASS_NUM):
    X = {}
    Y = {}
    for dataset in ['test', 'train', 'valid']:
        X[dataset] = []
        Y[dataset] = []
        with open('./MNIST/digits{0}.txt'.format(dataset)) as file:
            for line in file:
                parts = line.split(',')
                image = [float(i) for i in parts[:-1]]
                label = int(parts[-1])
                y = np.zeros(CLASS_NUM)
                y[label] = 1.0
                X[dataset].append(image)
                Y[dataset].append(y)
        X[dataset] = np.array(X[dataset])
        Y[dataset] = np.array(Y[dataset])
    return X, Y


def input_shakespeare(input_file='tiny-shakespeare/tiny-shakespeare.h5', seq_len=10, verbose=False):
    Data = {}
    indeces = set()
    with h5py.File(input_file, 'r') as f:
        # print('List of arrays in this file: \n', f.keys())
        Data['train'] = f.get('train')
        Data['val'] = f.get('val')
        Data['test'] = f.get('test')
        Data['train'] = np.array(Data['train'])
        Data['val'] = np.array(Data['val'])
        Data['test'] = np.array(Data['test'])
        if verbose:
            print('Shape of the training set is : ', Data['train'].shape)
            print('Shape of the validation set is : ', Data['val'].shape)
            print('Shape of the testing set is : ', Data['test'].shape)

    samples = {}
    samples['train'] = (Data['train'].shape[0]-1)/seq_len
    samples['val'] = (Data['val'].shape[0]-1)/seq_len
    samples['test'] = (Data['test'].shape[0]-1)/seq_len
    if verbose:
        print('train_samples is {0}'.format(samples['train']))
        print('val_samples is {0}'.format(samples['val']))
        print('test_samples is {0}'.format(samples['test']))

    for split in ['train', 'val', 'test']:
        for char in Data[split]:
            indeces.add(char)
    total_indeces = len(indeces)
    if verbose:
        print('Total number of indeces is {0}'.format(total_indeces))

    Vec_X, Vec_Y = {}, {}
    input_dim = total_indeces
    for split in ['train', 'val', 'test']:
        Vec_X[split] = []
        Vec_Y[split] = []
        vs_x, vs_y = [], []
        for i, char in enumerate(Data[split]):
            v = np.zeros(input_dim)
            v[int(char)-1] = 1.0
            if i < Data[split].shape[0] - 1:
                vs_x.append(v.T)
            if i > 0:
                vs_y.append(v.T)
            if len(vs_x) == seq_len:
                Vec_X[split].append(np.array(vs_x).T)
                vs_x = []
            if len(vs_y) == seq_len:
                Vec_Y[split].append(np.array(vs_y).T)
                vs_y = []
        Vec_X[split] = np.array(Vec_X[split])
        Vec_Y[split] = np.array(Vec_Y[split])
        Vec_X[split] = np.swapaxes(Vec_X[split], 0, 2)
        Vec_Y[split] = np.swapaxes(Vec_Y[split], 0, 2)
        assert Vec_X[split].shape == Vec_Y[split].shape == (seq_len, input_dim, samples[split]), '[Error] Vec[{0}] shape is ' \
                                                                                                        'wrong'.format(split)

    X, Y = {}, {}
    for split in ['train', 'val', 'test']:
        X[split], Y[split] = {}, {}
        for t in range(seq_len):
            X[split][t] = Vec_X[split][t, :, :].reshape((input_dim, samples[split]))
            Y[split][t] = Vec_Y[split][t, :, :].reshape((input_dim, samples[split]))

    return X, Y, samples, input_dim


def plot_error(train_error, validate_error, figname):
    fig = plt.figure()
    n_epoch = len(train_error)
    x = range(1, n_epoch+1)
    te, = plt.plot(x, train_error, label='training error')
    ve, = plt.plot(x, validate_error, label='validation error')
    plt.legend(handles=[te, ve])
    plt.show()
    fig.savefig('{0}.pdf'.format(figname))

def add_noise(X, threshold=0.75):
    n_samples = X.shape[0]
    input_dim = X.shape[1]
    noise = np.random.uniform(0.0, 1.0, size=X.shape)
    X_noise = np.array([[0.0 if noise[r,i] > threshold else X[r,i] for i in range(input_dim)] for r in range(n_samples)])
    assert X_noise.shape == X.shape, '[Error] X_noise shape is wrong'
    return X_noise

def shuffle_data(X, Y):
    X_train = X['train']
    Y_train = Y['train']
    indices = np.random.permutation(X_train.shape[0])
    X_train = X_train[indices]
    Y_train = Y_train[indices]
    return X_train, Y_train

def dropout( X, p = 0. ):
    if p != 0:
        retain_p = 1 - p
        X = X * np.random.binomial(1,retain_p,size = X.shape)
        X /= retain_p
    return X