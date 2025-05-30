import numpy as np
from scipy.optimize import minimize

import datasets
import models
import compute_entropy
import utils_ssl
import utils_sampling


def bo_ssl_sampling(str_model, bounds, seed, ind_iter, X, by, info_sampling, gamma=None):
    assert isinstance(gamma, (type(None), float))

    method_sampling, num_sampling, num_test = utils_sampling.get_info_sampling(info_sampling)
    X_tran, X_test = utils_sampling.get_samples(bounds, method_sampling, num_sampling, seed, ind_iter, num_test=num_test, X=X)

    X, by, labels, weights, val_to_split, labels_original = datasets.get_dataset(X, by)
    assert np.all(labels == labels_original)

    # gaussian_0 and uniform_0 works, but sobol_0 and halton_0 do not work.
    if X_tran.shape[0] > 0:
        X = np.concatenate([X, X_tran], axis=0)
        labels = np.concatenate([labels, -1 * np.ones(X_tran.shape[0])], axis=0)

    if gamma is None:
        gamma = compute_entropy.optimize_gamma(str_model, X, labels)

    model = models.get_model(str_model, gamma=gamma)
    model.fit(X, labels)

    preds = model.predict_proba(X_test)

    preds = preds[:, 1]
    preds[np.isnan(preds)] = 0.0

    indices = np.where(preds >= (1.0 - 1e-2))[0]

    if indices.shape[0] > 0:
        ind_min = np.random.choice(indices)
        next_point = X_test[ind_min]
    else:
        def fun(bx):
            pred = model.predict_proba([bx])[0, 1]

            if np.isnan(pred):
                pred = 0.0

            return pred

        fun_optimize = lambda bx: -1.0 * fun(bx)

        top_k = 100
        X_test_ = X_test[np.argpartition(preds, -top_k)[-top_k:]]
        acquisitions = []
        acquisition_vals = []

        for bx_test in X_test_:
            result = minimize(fun_optimize, x0=bx_test, method='L-BFGS-B',
                bounds=bounds, jac=False,
                options={'maxiter': 1000, 'ftol': 1e-9})

            bx_acq = result.x
            val_acq = result.fun

            acquisitions.append(bx_acq)
            acquisition_vals.append(val_acq)

        ind_min = utils_ssl.get_minimum_index(np.array(acquisition_vals))
        next_point = acquisitions[ind_min]

    return next_point, model, labels_original, gamma


if __name__ == '__main__':
    from bayeso_benchmarks import utils as bb_utils

    obj_target = bb_utils.get_benchmark('branin')
    bounds = obj_target.get_bounds()

    X = obj_target.sample_uniform(10, seed=42)
    by = obj_target.output(X)
    by = np.squeeze(by, axis=1)

    seed = 42

    bo_ssl_sampling('label_propagation', bounds, seed, 10, X, by, 'uniform_100')
    bo_ssl_sampling('label_propagation', bounds, seed, 10, X, by, 'halton_1000')

    bo_ssl_sampling('label_spreading', bounds, seed, 10, X, by, 'uniform_100')
    bo_ssl_sampling('label_spreading', bounds, seed, 10, X, by, 'halton_1000')
