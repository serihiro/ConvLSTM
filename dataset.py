import os

import chainer
import numpy as np


class MovingMnistDataset(chainer.dataset.DatasetMixin):
    def __init__(self, l, r, inn, outn, path="./mnist_test_seq.npy"):
        self.l = l
        self.r = r
        self.inn = inn
        self.outn = outn
        self.data = np.load(path)
        self.data[self.data < 128] = 0
        self.data[self.data >= 128] = 1

    def __len__(self):
        return self.r - self.l

    def get_example(self, i):
        ind = self.l + i
        return self.data[:self.inn, ind, :, :].astype(np.int32), self.data[self.inn:self.inn + self.outn, ind, :,
                                                                 :].astype(np.int32)


class JmaGpvDataset(chainer.dataset.DatasetMixin):
    def __init__(self, index_file_path, n_in, n_out, root_path=".", threshold=5.0):
        with open(os.path.join(root_path, index_file_path), mode='r') as f:
            self._path_list = f.read().split('\n')
        self._n_in = n_in
        self._n_out = n_out
        self._threshold = threshold

    def __len__(self):
        return len(self._path_list)

    def get_example(self, i):
        data = np.load(self._path_list[i])
        data[data < self._threshold] = 0
        data[data >= self._threshold] = 1

        return data[:self._n_in, 0, :, :].astype(np.int32), data[self._n_in:self._n_in + self._n_out, 0, :, :].astype(
            np.int32)
