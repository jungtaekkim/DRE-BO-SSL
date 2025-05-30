import numpy as np
from bayeso import bo

import minimax_tilting_sampler


def get_initials(bounds, num_init, seed):
    random_state = np.random.RandomState(seed)
    dim = bounds.shape[0]

    initials = random_state.uniform(0, 1, (num_init, dim))
    initials = (bounds[:, 1] - bounds[:, 0]) * initials + bounds[:, 0]

    return initials


def get_info_sampling(info_sampling):
    list_info_sampling = info_sampling.split('_')
    assert len(list_info_sampling) == 2 or len(list_info_sampling) == 3

    method_sampling = str(list_info_sampling[0])
    num_sampling = int(list_info_sampling[1])

    if len(list_info_sampling) == 3:
        num_test = int(list_info_sampling[2])
    else:
        num_test = 1000

    print(f'method_sampling {method_sampling} num_sampling {num_sampling} num_test {num_test}', flush=True)

    return method_sampling, num_sampling, num_test


def get_samples(bounds, method_sampling, num_sampling, seed, ind_iter, num_test=1000, X=None):
    model_bo = bo.BOwGP(bounds)
    seed_ = seed * (1 + ind_iter) + 1221

    if method_sampling in ['uniform', 'sobol', 'halton']:
        X_tran = model_bo.get_samples(method_sampling, num_samples=num_sampling, seed=seed_)
    elif method_sampling in ['gaussian']:
        assert X is not None

        num_X = X.shape[0]
        num_sampling_per_gaussian = np.ones(num_X) * (num_sampling // num_X)
        remainder = num_sampling % num_X

        for ind in range(0, remainder):
            num_sampling_per_gaussian[ind] += 1

        X_tran = []

        for ind_elem, elem in enumerate(zip(X, num_sampling_per_gaussian)):
            bx, cur_num = elem

            lb = bounds[:, 0]
            ub = bounds[:, 1]
            mvn = minimax_tilting_sampler.TruncatedMVN(bx, np.eye(bx.shape[0]), lb, ub, seed=(seed_ + ind_elem))

            for _ in range(0, int(cur_num)):
                bx_tran = mvn.sample(1)
                bx_tran = bx_tran.T
                bx_tran = bx_tran[0]
                assert np.all(bx_tran >= lb) and np.all(bx_tran <= ub)
                X_tran.append(bx_tran)

        X_tran = np.array(X_tran)
        assert X_tran.shape[0] == num_sampling
    else:
        raise ValueError

    X_test = model_bo.get_samples('uniform', num_samples=num_test, seed=seed_ + 123321)

    return X_tran, X_test
