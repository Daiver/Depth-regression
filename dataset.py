import h5py
import numpy as np
from scipy.io import loadmat


class NyuDepthDataset:

    @staticmethod
    def from_file(path_to_dataset, transform=None):
        f = h5py.File(path_to_dataset, 'r')
        n_samples = f['images'].shape[0]
        print('n samples', n_samples)

        images = f['images']
        images = np.moveaxis(images, [1, 2], [3, 2])
        depths = f['depths']
        depths = np.moveaxis(depths, 1, 2)
        depths = depths.reshape(depths.shape + (1,))

        return NyuDepthDataset(images, depths, transform)

    @staticmethod
    def from_file_with_splits(path_to_dataset, path_to_splits, transform=None):
        splits_mat = loadmat(path_to_splits)
        train_idxs = (splits_mat['trainNdxs'] - 1).reshape(-1)
        test_idxs = (splits_mat['testNdxs'] - 1).reshape(-1)

        dataset = NyuDepthDataset.from_file(path_to_dataset)
        train_dataset = NyuDepthDataset(dataset.images[train_idxs], dataset.depths[train_idxs], transform=transform)
        test_dataset = NyuDepthDataset(dataset.images[test_idxs], dataset.depths[test_idxs], transform=transform)
        return train_dataset, test_dataset

    def __init__(self, images, depths, transform=None):
        assert len(images) == len(depths)
        self.images = images
        self.depths = depths
        self.transform = transform

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        if self.transform is None:
            return self.images[idx], self.depths[idx]
        return self.transform(self.images[idx], self.depths[idx])
