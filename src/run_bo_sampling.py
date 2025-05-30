import os
import numpy as np
import time
import argparse

from bayeso_benchmarks import utils as bb_utils

import ssl_sampling
import utils_bayesopt
import utils_sampling
import utils_printing


str_task = 'sampling'
path_results = f'../results_{str_task}'


def run_bo(str_model, obj_target, num_init, num_iter, seed, info_sampling):
    assert info_sampling is not None

    bounds = obj_target.get_bounds()
    print('bounds')
    print(bounds)

    X = utils_sampling.get_initials(bounds, num_init, seed)
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

        gamma = None

        next_point, _, _, gamma_returned = ssl_sampling.bo_ssl_sampling(
            str_model, bounds, seed, ind_iter,
            X, np.array(by), info_sampling, gamma
        )

        gammas.append(gamma_returned)

        y = utils_bayesopt.get_output(obj_target.output, next_point)
        X = np.concatenate([X, [next_point]], axis=0)
        by.append(y)

        time_end = time.time()
        time_end_start = time_end - time_start
        times.append(time_end_start)

        print(f'Iteration {ind_iter+1:03d}: cur_best {np.min(by):.4f}')
        if (ind_iter + 1) % 100 == 0 and (ind_iter + 1) < num_iter:
            print('')

    X = np.array(X)
    by = np.array(by)
    times = np.array(times)
    assert X.shape[0] == by.shape[0] == times.shape[0]

    print('')
    utils_printing.print_separators()
    print(f'X.shape {X.shape} by.shape {by.shape} times.shape {times.shape}')
    print(f'final_best {np.min(by):.4f}')
    utils_printing.print_separators()
    print('', flush=True)

    dict_info = {
        'str_model': str_model,
        'info_sampling': info_sampling,
        'str_target': obj_target.name,
        'num_init': num_init,
        'num_iter': num_iter,
        'seed': seed,
        'X': X,
        'by': by,
        'times': times,
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
    parser.add_argument('--info_sampling', type=str, required=True)

    args = parser.parse_args()

    str_model = args.model
    str_target = args.target
    seed = args.seed
    num_init = args.num_init
    num_iter = args.num_iter
    info_sampling = args.info_sampling

    utils_printing.print_info(str_model, str_target, seed, num_init, num_iter, info_sampling)

    if True:
        list_str_target = str_target.split('_')
        if len(list_str_target) == 1:
            obj_target = bb_utils.get_benchmark(list_str_target[0])
        elif len(list_str_target) == 2:
            obj_target = bb_utils.get_benchmark(list_str_target[0], dim=int(list_str_target[1]))
        else:
            raise ValueError

    run_bo(str_model, obj_target, num_init, num_iter, seed, info_sampling)
