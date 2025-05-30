import numpy as np


class DigitsMNIST:
    def __init__(self):
        self.name = f'digits_mnist'

        # You should change it after downloading this file.
        self.str_dataset = '../../datasets/mnist_test.npy'
        file_dataset = np.load(self.str_dataset, allow_pickle=True)
        file_dataset = file_dataset[()]

        self.X_pool = file_dataset['Z_test']
        self.by_pool = file_dataset['by_test']

    def convert_to_index(self, bx):
        ind = np.argmin(np.linalg.norm(self.X_pool - bx, axis=1, ord=2))
        return ind

    def get_bounds(self):
        bounds = np.array([
            np.min(self.X_pool, axis=0),
            np.max(self.X_pool, axis=0),
        ]).T

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
        ind_bx = self.convert_to_index(bx)

        y = float(self.by_pool[ind_bx])
        y /= 1000.0

        return y


if __name__ == '__main__':
    obj = DigitsMNIST()

    bx = obj.sample_uniform(1)[0]
    print(bx)
    output = obj.output(bx)
    print(output)
