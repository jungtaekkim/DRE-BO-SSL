import numpy as np


def get_minimum_index(vals, seed=None):
    min_vals = np.min(vals)
    ind_min = np.random.RandomState(seed).choice(np.where((vals - min_vals) < 1e-8)[0])

    return ind_min
