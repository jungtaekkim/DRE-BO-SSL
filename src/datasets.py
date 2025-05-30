import numpy as np
import torch
from torch.utils.data import Dataset


def get_dataset(X, by, use_weights, gamma=0.33):
    try:
        val_to_split = np.quantile(by, gamma, method='nearest')
    except:
        val_to_split = np.quantile(by, gamma, interpolation='nearest')
    labels = (by <= val_to_split).astype(float)
    labels_original = labels

    if not use_weights:
        X = X
        by = by

        weights = np.ones_like(by)
    else:
        bool_1 = labels == 1

        X_1 = X[bool_1]
        by_1 = by[bool_1]
        labels_1 = labels[bool_1]

        X_0 = X
        by_0 = by
        labels_0 = np.zeros_like(labels)

        bw_1 = (val_to_split - by)[bool_1]
        if np.mean(bw_1) > 0.0:
            bw_1 = bw_1 / np.mean(bw_1)
        else:
            # it will be a uniform distribution.
            assert np.all(bw_1 == 0.0)
            bw_1 += 1e-12
            bw_1 = bw_1 / np.mean(bw_1)
        bw_0 = 1.0 - labels_0

        print(X.shape, X_1.shape, by_1.shape, labels_1.shape)

        X = np.concatenate([X_1, X_0], axis=0)
        by = np.concatenate([by_1, by_0], axis=0)
        labels = np.concatenate([labels_1, labels_0], axis=0)
        print(X.shape, by.shape, labels.shape)

        s_1 = X_1.shape[0]
        s_0 = X_0.shape[0]

        weights = np.concatenate([bw_1 * (s_1 + s_0) / s_1, bw_0 * (s_1 + s_0) / s_0], axis=0)
        weights = weights / np.mean(weights)

    return X, by, labels, weights, val_to_split, labels_original


class BODataset(Dataset):
    def __init__(self, X, by, labels, weights, val_to_split):
        assert len(X.shape) == 2
        assert len(by.shape) == 1
        assert len(labels.shape) == 1
        assert len(weights.shape) == 1
        assert X.shape[0] == by.shape[0] == labels.shape[0] == weights.shape[0]
        assert isinstance(val_to_split, float)

        self.X = torch.tensor(X).float()
        self.by = torch.tensor(by).float()
        self.labels = torch.tensor(labels).float()
        self.weights = torch.tensor(weights).float()
        self.val_to_split = val_to_split

        self.num_X = X.shape[0]

    def __len__(self):
        return self.num_X

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'X': self.X[idx], 'by': self.by[idx], 'weights': self.weights[idx], 'labels': self.labels[idx]}

        return sample


if __name__ == '__main__':
    import numpy as np
    from torch.utils.data import DataLoader

    dataset = BODataset(np.random.randn(111, 4), np.random.randn(111), np.random.randn(111), np.random.randn(111), 0.2)

    print(dataset.X)
    print(dataset.labels)

    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

    for batch in dataloader:
        print(batch)
        print(batch['X'].shape, batch['by'].shape, batch['labels'].shape, batch['weights'].shape)

    for batch in dataloader:
        print(batch)
        print(batch['X'].shape, batch['by'].shape, batch['labels'].shape, batch['weights'].shape)
