import numpy as np
from nats_bench import create


possible_choices = [8, 16, 24, 32, 40, 48, 56, 64]
bound = [
    possible_choices[0] - (possible_choices[1] - possible_choices[0]) / 2,
    possible_choices[-1] + (possible_choices[-1] - possible_choices[-2]) / 2,
]
len_possible_choices = len(possible_choices)
num_hyperparameters = 5

# You should change it after downloading this file.
path_dataset = '../../datasets/NATS-sss-v1_0-50262-simple'
api = create(path_dataset, 'sss', fast_mode=True, verbose=False)


def transform_choices_to_hyperparameters(choices):
    assert isinstance(choices, np.ndarray)
    assert choices.shape[0] == num_hyperparameters

    hyperparameters = []

    for choice in choices:
        hyperparameters.append(possible_choices[choice])
    hyperparameters = np.array(hyperparameters)

    return hyperparameters


def transform_continuous_to_discrete(inputs):
    assert isinstance(inputs, np.ndarray)
    assert inputs.shape[0] == num_hyperparameters

    hyperparameters = []

    for elem in inputs:
        ind_min = np.argmin(np.abs(possible_choices - elem))
        hyperparameters.append(possible_choices[ind_min])
    hyperparameters = np.array(hyperparameters)

    return hyperparameters


def get_index(hyperparameters):
    assert isinstance(hyperparameters, np.ndarray)
    assert hyperparameters.shape[0] == num_hyperparameters

    out_channel_of_1st_conv_layer = hyperparameters[0]
    out_channel_of_1st_cell_stage = hyperparameters[1]
    out_channel_of_1st_residual_block = hyperparameters[2]
    out_channel_of_2nd_cell_stage = hyperparameters[3]
    out_channel_of_2nd_residual_block = hyperparameters[4]

    arch = f'{out_channel_of_1st_conv_layer}:{out_channel_of_1st_cell_stage}:{out_channel_of_1st_residual_block}:{out_channel_of_2nd_cell_stage}:{out_channel_of_2nd_residual_block}'

    index = api.query_index_by_arch(arch)

    return index


def get_single_valid(dataset, index):
    assert isinstance(dataset, str)
    assert isinstance(index, int)
    assert dataset in ['cifar10-valid', 'cifar100', 'ImageNet16-120']

    info = api.get_more_info(index, dataset, is_random=True)
    acc_valid = info['valid-accuracy']
    err_valid = (100.0 - acc_valid) / 100.0

    return acc_valid, err_valid


def get_average_valid_test(dataset, index):
    assert isinstance(dataset, str)
    assert isinstance(index, int)
    assert dataset in ['cifar10-valid', 'cifar100', 'ImageNet16-120']

    info = api.get_more_info(index, dataset, is_random=False)
    acc_valid = info['valid-accuracy']
    err_valid = (100.0 - acc_valid) / 100.0

    acc_test = info['test-accuracy']
    err_test = (100.0 - acc_test) / 100.0

    return acc_valid, err_valid, acc_test, err_test


def get_best_precomputed(dataset, valid_or_test):
    assert isinstance(dataset, str)
    assert isinstance(valid_or_test, str)
    assert dataset in ['cifar10-valid', 'cifar100', 'ImageNet16-120']
    assert valid_or_test in ['valid', 'test']

    if dataset == 'cifar10-valid':
        if valid_or_test == 'valid':
            err_best = 0.1497999998535157
        elif valid_or_test == 'test':
            err_best = 0.15159999999999996
    elif dataset == 'cifar100':
        if valid_or_test == 'valid':
            err_best = 0.3893999997558593
        elif valid_or_test == 'test':
            err_best = 0.3888000004272461
    elif dataset == 'ImageNet16-120':
        if valid_or_test == 'valid':
            err_best = 0.6086666679890951
        elif valid_or_test == 'test':
            err_best = 0.6053333338419596

    return err_best


class NATSBench:
    def __init__(self, dataset, num_hyperparameters=num_hyperparameters):
        self.dataset = dataset
        self.num_hyperparameters = num_hyperparameters

        global_minimum = get_best_precomputed(self.dataset, 'test')
        self.global_minimum = global_minimum

        self.name = f'natsbench_{self.dataset}'

        X_pool = []

        for elem_1 in range(0, len_possible_choices):
            for elem_2 in range(0, len_possible_choices):
                for elem_3 in range(0, len_possible_choices):
                    for elem_4 in range(0, len_possible_choices):
                        for elem_5 in range(0, len_possible_choices):
                            X_pool.append([
                                possible_choices[elem_1],
                                possible_choices[elem_2],
                                possible_choices[elem_3],
                                possible_choices[elem_4],
                                possible_choices[elem_5],
                            ])

        self.X_pool = np.array(X_pool)

    def get_bounds(self):
        bounds = [
            bound
        ] * self.num_hyperparameters
        bounds = np.array(bounds)

        return bounds

    def validate(self, bx):
        assert isinstance(bx, np.ndarray)
        assert len(bx.shape) == 1
        assert bx.shape[0] == self.num_hyperparameters

        bounds = self.get_bounds()
        assert np.all(bx >= bounds[:, 0]) and np.all(bx <= bounds[:, 1])
        return bx

    def sample_uniform(self, num_points, seed=None):
        assert isinstance(num_points, int)
        assert isinstance(seed, (type(None), int))

        random_state_ = np.random.RandomState(seed)
        dim_problem = self.num_hyperparameters

        bounds = self.get_bounds()

        points = random_state_.uniform(size=(num_points, dim_problem))
        points = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * points

        return points

    def output(self, bx):
        bx = self.validate(bx)
        hyperparameters = transform_continuous_to_discrete(bx)
        index = get_index(hyperparameters)
        acc_valid, err_valid, _, _ = get_average_valid_test(self.dataset, index)

        return err_valid

    def final_output(self, bx):
        bx = self.validate(bx)
        hyperparameters = transform_continuous_to_discrete(bx)
        index = get_index(hyperparameters)
        acc_valid, err_valid, acc_test, err_test = get_average_valid_test(self.dataset, index)

        return err_test


if __name__ == '__main__':
    obj = NATSBench('cifar10-valid')

    print(obj.get_bounds())
    print(obj.sample_uniform(10))
    print(obj.output(np.array([14.2, 50.9, 24.2, 29.0, 60.1])))
    print(obj.final_output(np.array([14.2, 50.9, 24.2, 29.0, 60.1])))
    print(obj.global_minimum)

    obj = NATSBench('cifar100')
    print(obj.global_minimum)

    obj = NATSBench('ImageNet16-120')
    print(obj.global_minimum)
