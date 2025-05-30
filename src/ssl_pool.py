import numpy as np

import models
import datasets
import compute_entropy
import utils_ssl


def bo_ssl_pool(str_model, str_target, X, by, X_pool, seed, gamma=None, num_subset=2000):
    assert isinstance(gamma, (type(None), float))

    indices_pool = np.full(X_pool.shape[0], True)

    for bx in X:
        indices_pool = np.logical_and(indices_pool, ~np.all(X_pool == bx, axis=1))
    X_pool_ = X_pool[indices_pool]

    X, by, labels, weights, val_to_split, labels_original = datasets.get_dataset(X, by)
    assert np.all(labels == labels_original)

    num_subset = np.minimum(num_subset, X_pool_.shape[0])
    ind_subset = np.random.RandomState(seed + 111).choice(X_pool_.shape[0], num_subset, replace=False)
    X_pool_subset = X_pool_[ind_subset]

    X = np.concatenate([X, X_pool_subset], axis=0)
    labels = np.concatenate([labels, -1 * np.ones(X_pool_subset.shape[0])], axis=0)

    if str_target in [
        'natsbench_cifar10-valid',
        'natsbench_cifar100',
        'natsbench_ImageNet16-120',
        'tabularbenchmarks_protein',
        'tabularbenchmarks_slice',
        'tabularbenchmarks_naval',
        'tabularbenchmarks_parkinsons',
        'digits_mnist',
    ]:
        bounds = np.array([[1e-5, 5e-3]])
    else:
        bounds = np.array([[1e-5, 1e5]])

    if gamma is None:
        gamma = compute_entropy.optimize_gamma(str_model, X, labels, bounds=bounds)

    model = models.get_model(str_model, gamma=gamma)
    model.fit(X, labels)

    preds = []

    num_gap = 100
    num_splits = X_pool_.shape[0] // num_gap

    if X_pool_.shape[0] % num_gap != 0:
        num_splits += 1

    for ind_pool in range(0, num_splits):
        preds_ = model.predict_proba(X_pool_[ind_pool * num_gap:(ind_pool + 1) * num_gap])
        preds_ = preds_[:, 1]
        preds_[np.isnan(preds_)] = 0.0
        preds += list(preds_)

    preds = np.array(preds)
    assert len(preds.shape) == 1
    assert X_pool_.shape[0] == preds.shape[0]

    indices = np.where(preds >= (1.0 - 1e-2))[0]

    if indices.shape[0] > 0:
        ind_min = np.random.RandomState(seed + 1001).choice(indices)
    else:
        ind_min = utils_ssl.get_minimum_index(-1.0 * preds, seed=seed)

    next_point = X_pool_[ind_min]
    return next_point, model, labels_original, gamma


if __name__ == '__main__':
    from bayeso_benchmarks import utils as bb_utils

    obj_target = bb_utils.get_benchmark('branin')

    X = obj_target.sample_uniform(10, seed=42)
    by = obj_target.output(X)
    by = np.squeeze(by, axis=1)
    X_pool = obj_target.sample_uniform(100, seed=42)

    bo_ssl_pool('label_propagation', 'branin', X, by, X_pool, 42)
    bo_ssl_pool('label_spreading', 'branin', X, by, X_pool, 42)
