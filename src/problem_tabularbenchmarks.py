import numpy as np
import ConfigSpace
import tabular_benchmarks


class TabularBenchmarks:
    def __init__(self, dataset):
        self.dataset = dataset

        # You should change it after downloading this file.
        path_datasets = '../../datasets/fcnet_tabular_benchmarks'

        if dataset == 'protein':
            model = tabular_benchmarks.FCNetProteinStructureBenchmark(data_dir=path_datasets, seed=42)
        elif dataset == 'slice':
            model = tabular_benchmarks.FCNetSliceLocalizationBenchmark(data_dir=path_datasets, seed=42)
        elif dataset == 'naval':
            model = tabular_benchmarks.FCNetNavalPropulsionBenchmark(data_dir=path_datasets, seed=42)
        elif dataset == 'parkinsons':
            model = tabular_benchmarks.FCNetParkinsonsTelemonitoringBenchmark(data_dir=path_datasets, seed=42)
        else:
            raise ValueError

        self.model = model
        self.config_space = self.model.get_configuration_space()

        self.name = f'tabularbenchmarks_{self.dataset}'

        self.list_n_units_1 = [16, 32, 64, 128, 256, 512]
        self.list_n_units_2 = [16, 32, 64, 128, 256, 512]
        self.list_dropout_1 = [0.0, 0.3, 0.6]
        self.list_dropout_2 = [0.0, 0.3, 0.6]
        self.list_activation_fn_1 = ["tanh", "relu"]
        self.list_activation_fn_2 = ["tanh", "relu"]
        self.list_init_lr = [5 * 1e-4, 1e-3, 5 * 1e-3, 1e-2, 5 * 1e-2, 1e-1]
        self.list_lr_schedule = ["cosine", "const"]
        self.list_batch_size = [8, 16, 32, 64]

        X_pool = []

        for ind_elem_1, _ in enumerate(self.list_n_units_1):
            for ind_elem_2, _ in enumerate(self.list_n_units_2):
                for ind_elem_3, _ in enumerate(self.list_dropout_1):
                    for ind_elem_4, _ in enumerate(self.list_dropout_2):
                        for ind_elem_5, _ in enumerate(self.list_activation_fn_1):
                            for ind_elem_6, _ in enumerate(self.list_activation_fn_2):
                                for ind_elem_7, _ in enumerate(self.list_init_lr):
                                    for ind_elem_8, _ in enumerate(self.list_lr_schedule):
                                        for ind_elem_9, _ in enumerate(self.list_batch_size):
                                            X_pool.append([
                                                ind_elem_1,
                                                ind_elem_2,
                                                ind_elem_3,
                                                ind_elem_4,
                                                ind_elem_5,
                                                ind_elem_6,
                                                ind_elem_7,
                                                ind_elem_8,
                                                ind_elem_9,
                                            ])

        self.X_pool = np.array(X_pool).astype(float)

    def convert_to_index(self, bx):
        new_bx = [
            int(bx[0]),
            int(bx[1]),
            int(bx[2]),
            int(bx[3]),
            int(bx[4]),
            int(bx[5]),
            int(bx[6]),
            int(bx[7]),
            int(bx[8]),
        ]

        return np.array(new_bx)

    def get_config(self, bx):
        bx = self.convert_to_index(bx)

        dict_config = {
            'n_units_1': self.list_n_units_1[bx[0]],
            'n_units_2': self.list_n_units_2[bx[1]],
            'dropout_1': self.list_dropout_1[bx[2]],
            'dropout_2': self.list_dropout_2[bx[3]],
            'activation_fn_1': self.list_activation_fn_1[bx[4]],
            'activation_fn_2': self.list_activation_fn_2[bx[5]],
            'init_lr': self.list_init_lr[bx[6]],
            'lr_schedule': self.list_lr_schedule[bx[7]],
            'batch_size': self.list_batch_size[bx[8]],
        }

        config = ConfigSpace.configuration_space.Configuration(self.config_space, values=dict_config)

        return config

    def get_bounds(self):
        bounds = np.array([
            [0, len(self.list_n_units_1) - 1 + 0.999],
            [0, len(self.list_n_units_2) - 1 + 0.999],
            [0, len(self.list_dropout_1) - 1 + 0.999],
            [0, len(self.list_dropout_2) - 1 + 0.999],
            [0, len(self.list_activation_fn_1) - 1 + 0.999],
            [0, len(self.list_activation_fn_2) - 1 + 0.999],
            [0, len(self.list_init_lr) - 1 + 0.999],
            [0, len(self.list_lr_schedule) - 1 + 0.999],
            [0, len(self.list_batch_size) - 1 + 0.999],
        ])

        return bounds

    def validate(self, bx):
        assert isinstance(bx, np.ndarray)
        assert len(bx.shape) == 1

        bounds = self.get_bounds()
        assert np.all(bx >= bounds[:, 0]) and np.all(bx <= bounds[:, 1])
        return bx

    def sample_uniform(self, num_points, seed=None):
        assert isinstance(num_points, int)
        assert isinstance(seed, (type(None), int))

        random_state_ = np.random.RandomState(seed)

        bounds = self.get_bounds()
        dim_problem = bounds.shape[0]

        points = random_state_.uniform(size=(num_points, dim_problem))
        points = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * points

        return points

    def output(self, bx):
        bx = self.validate(bx)
        config = self.get_config(bx)

        y, cost = self.model.objective_function(config)
        y = float(y)

        return y


if __name__ == '__main__':
    obj = TabularBenchmarks('protein')

    bx = obj.sample_uniform(1)[0]
    print(bx)
    config = obj.get_config(bx)
    print(config)
    output = obj.output(bx)
    print(output)
    obj = TabularBenchmarks('protein')
    output = obj.output(bx)
    print(output)

    obj = TabularBenchmarks('slice')

    bx = obj.sample_uniform(1)[0]
    print(bx)
    config = obj.get_config(bx)
    print(config)
    output = obj.output(bx)
    print(output)
    obj = TabularBenchmarks('slice')
    output = obj.output(bx)
    print(output)

    obj = TabularBenchmarks('naval')

    bx = obj.sample_uniform(1)[0]
    print(bx)
    config = obj.get_config(bx)
    print(config)
    output = obj.output(bx)
    print(output)
    obj = TabularBenchmarks('naval')
    output = obj.output(bx)
    print(output)

    obj = TabularBenchmarks('parkinsons')

    bx = obj.sample_uniform(1)[0]
    print(bx)
    config = obj.get_config(bx)
    print(config)
    output = obj.output(bx)
    print(output)
    obj = TabularBenchmarks('parkinsons')
    output = obj.output(bx)
    print(output)
