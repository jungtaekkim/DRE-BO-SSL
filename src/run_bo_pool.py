import os
import numpy as np
import time
import argparse

from bayeso_benchmarks import utils as bb_utils

import ssl_pool
import utils_bayesopt
import utils_sampling
import utils_printing


str_task = 'pool'
path_results = f'../results_{str_task}'


def choose_from_pool(point_to_compare, X_pool):
    assert len(point_to_compare.shape) == 1
    assert len(X_pool.shape) == 2
    assert point_to_compare.shape[0] == X_pool.shape[1]

    dist = np.linalg.norm(X_pool - point_to_compare, axis=1, ord=2)
    ind_min = np.argmin(dist)
    new_point = X_pool[ind_min]

    return new_point


def run_bo(str_model, obj_target, num_init, num_iter, X_pool, seed):
    bounds = obj_target.get_bounds()
    print('bounds', flush=True)
    print(bounds, flush=True)

    X = utils_sampling.get_initials(bounds, num_init, seed)
    new_X = []
    for bx in X:
        new_X.append(choose_from_pool(bx, X_pool))
    X = np.array(new_X)

    by = []
    times = []

    for bx in X:
        time_start = time.time()
        y = utils_bayesopt.get_output(obj_target.output, bx)
        by.append(y)

        time_end = time.time()
        time_end_start = time_end - time_start
        times.append(time_end_start)

    gammas = []

    for ind_iter in range(0, num_iter):
        time_start = time.time()

        if obj_target.name in [
            'natsbench_cifar10-valid',
            'natsbench_cifar100',
            'natsbench_ImageNet16-120',
        ]:
            num_subset = 2000
        elif obj_target.name in [
            'tabularbenchmarks_protein',
            'tabularbenchmarks_slice',
            'tabularbenchmarks_naval',
            'tabularbenchmarks_parkinsons',
        ]:
            num_subset = 2000
        elif obj_target.name in [
            'digits_mnist',
        ]:
            num_subset = 2000
        else:
            num_subset = 2000

        gamma = None

        next_point, _, _, gamma_returned = ssl_pool.bo_ssl_pool(
            str_model, obj_target.name, X, np.array(by), X_pool, seed,
            gamma=gamma,
            num_subset=num_subset,
        )

        gammas.append(gamma_returned)

        print(next_point, flush=True)

        random_state = np.random.RandomState(seed + 1002)
        while np.sum(np.all(next_point == X, axis=1)) > 0:
            ind_chosen = random_state.choice(X_pool.shape[0])
            next_point = X_pool[ind_chosen]

        print(next_point, flush=True)
        assert np.sum(np.all(X_pool == next_point, axis=1)) == 1

        y = utils_bayesopt.get_output(obj_target.output, next_point)
        X = np.concatenate([X, [next_point]], axis=0)
        by.append(y)

        time_end = time.time()
        time_end_start = time_end - time_start
        times.append(time_end_start)

        print(f'Iteration {ind_iter+1:03d}: cur_best {np.min(by):.4f}', flush=True)
        if (ind_iter + 1) % 100 == 0 and (ind_iter + 1) < num_iter:
            print('', flush=True)

    X = np.array(X)
    by = np.array(by)
    times = np.array(times)
    assert X.shape[0] == by.shape[0] == times.shape[0]

    for bx in X:
        assert np.sum(np.all(bx == X_pool, axis=1)) > 0

    print('', flush=True)
    utils_printing.print_separators()
    print(f'X.shape {X.shape} by.shape {by.shape} times.shape {times.shape}', flush=True)
    print(f'final_best {np.min(by):.4f}', flush=True)
    utils_printing.print_separators()
    print('', flush=True)

    dict_info = {
        'str_model': str_model,
        'str_target': obj_target.name,
        'num_init': num_init,
        'num_iter': num_iter,
        'seed': seed,
        'X': X,
        'by': by,
        'times': times,
        'size_pool': X_pool.shape[0],
        'X_pool': X_pool,
        'gammas': np.array(gammas),
    }
    utils_bayesopt.save_bo(path_results, dict_info, str_task)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--num_init', type=int, required=True)
    parser.add_argument('--num_iter', type=int, required=True)

    args = parser.parse_args()

    str_model = args.model
    str_target = args.target
    seed = args.seed
    num_init = args.num_init
    num_iter = args.num_iter

    utils_printing.print_info(str_model, str_target, seed, num_init, num_iter, None)

    if str_target in ['natsbench_cifar10-valid', 'natsbench_cifar100', 'natsbench_ImageNet16-120']:
        import problem_natsbench

        list_str_target = str_target.split('_')
        obj_target = problem_natsbench.NATSBench(list_str_target[1])

        X_pool = obj_target.X_pool
    elif str_target in ['tabularbenchmarks_protein', 'tabularbenchmarks_slice', 'tabularbenchmarks_naval', 'tabularbenchmarks_parkinsons']:
        import problem_tabularbenchmarks

        list_str_target = str_target.split('_')
        obj_target = problem_tabularbenchmarks.TabularBenchmarks(list_str_target[1])

        X_pool = obj_target.X_pool
    elif str_target in ['digits_mnist']:
        import problem_mnist

        obj_target = problem_mnist.DigitsMNIST()

        X_pool = obj_target.X_pool
    else:
        list_str_target = str_target.split('_')
        if len(list_str_target) == 1:
            obj_target = bb_utils.get_benchmark(list_str_target[0])
        elif len(list_str_target) == 2:
            obj_target = bb_utils.get_benchmark(list_str_target[0], dim=int(list_str_target[1]))
        else:
            raise ValueError

        num_pool = 1000
        X_pool = utils_sampling.get_initials(obj_target.get_bounds(), num_pool, seed + 10001)

    for bx in X_pool:
        assert np.sum(np.all(bx == X_pool, axis=1)) == 1

    run_bo(str_model, obj_target, num_init, num_iter, X_pool, seed)
