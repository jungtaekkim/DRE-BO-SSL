import numpy as np
from scipy.optimize import minimize

import models


def safe_log(X):
    locations = np.where(X > 0)
    log_X = np.zeros_like(X, dtype=np.float64)
    log_X[locations] = np.log(X[locations])
    return log_X


def compute_entropy(hC):
    assert len(hC.shape) == 2

    ent = np.sum(hC * safe_log(hC), axis=1)
    ent = np.mean(ent, axis=0)
    ent *= -1.0
    return ent


def optimize_gamma(str_model, X, labels, bounds=np.array([[1e-5, 1e5]])):
    def fun_entropy(gamma):
        if isinstance(gamma, np.ndarray):
            assert gamma.ndim == 1
            assert gamma.shape[0] == 1

            gamma = gamma[0]

        model = models.get_model(str_model, gamma=gamma)
        model.fit(X, labels)

        preds = []
        num_gap = 100
        num_splits = X.shape[0] // num_gap

        if X.shape[0] % num_gap != 0:
            num_splits += 1

        for ind_pool in range(0, num_splits):
            preds_ = model.predict_proba(X[ind_pool * num_gap:(ind_pool + 1) * num_gap])
            preds += list(preds_)

        preds = np.array(preds)
        assert len(preds.shape) == 2
        assert X.shape[0] == preds.shape[0]

        ent = compute_entropy(preds)

        return ent

    x0 = 10**((np.log10(bounds[0, 0]) + np.log10(bounds[0, 1])) / 2.0)
    result = minimize(fun_entropy, x0=x0, method='L-BFGS-B',
        bounds=bounds, jac=False,
        options={'maxiter': 1000, 'ftol': 1e-9})
    gamma_optimized = result.x[0]

    return gamma_optimized


if __name__ == '__main__':
    X = np.random.randint(10, size=(10, 2))
    print(X)
    print(np.log(X))
    print(safe_log(X))

    X = np.random.uniform(size=(10, 4))
    print(X)
    print(compute_entropy(X))

    X = np.random.uniform(size=(100, 4))
    labels = np.concatenate([
        np.random.randint(2, size=(60, )),
        np.ones(40) * -1,
    ], axis=0).astype(np.int64)

    gamma_optimized = optimize_gamma('label_propagation', X, labels)
    print('label_propagation', gamma_optimized)

    gamma_optimized = optimize_gamma('label_spreading', X, labels)
    print('label_spreading', gamma_optimized)
