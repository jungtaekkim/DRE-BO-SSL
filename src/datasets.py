import numpy as np


def get_dataset(X, by, gamma=0.33):
    try:
        val_to_split = np.quantile(by, gamma, method='nearest')
    except:
        val_to_split = np.quantile(by, gamma, interpolation='nearest')
    labels = (by <= val_to_split).astype(float)
    labels_original = labels

    weights = np.ones_like(by)

    return labels, weights, val_to_split, labels_original
