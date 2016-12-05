import numpy as np


def classification_accuracy(outputs, targets, derivative=False):
    assert not derivative, '[Error] accuracy can not be uased as cost function'
    acc = 0
    outputs = np.atleast_2d(outputs.T)
    targets = np.atleast_2d(targets.T)
    for output, target in zip(outputs, targets):
        out = list(output)
        pred = out.index(max(out))
        tar = list(target).index(target[target == 1])
        if pred == tar:
            acc += 1
    acc = float(acc) / float(len(outputs))
    return acc