import numpy as np
import chainer
import time
from datetime import datetime
import csv
from pathlib import Path
import sys

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


class JmaGpzDataset(chainer.dataset.DatasetMixin):
    def __init__(self, l, r, inn, outn, file="./jmagpz.npz", profile_get_example=False):
        self._profile_get_example = profile_get_example
        if profile_get_example:
            self._profile_get_example_csv_path = f'./profile_get_example_{datetime.now().strftime("%Y%m%d_%H%M%S%f")}.csv'
            Path(self._profile_get_example_csv_path).touch(exist_ok=True)
            with open(self._profile_get_example_csv_path, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(['index', 'elapsed_time', 'size', 'timestamp'])

        if (type(file) is list) & (len(file) > 1):
            self.data = JmaGpzDataset.MultipleFileData(l, r, inn, outn, file)
        else:
            _file = file[0] if type(file) is list else file
            self.data = JmaGpzDataset.SingleFileData(l, r, inn, outn, _file)

    def __len__(self):
        return self.data.__len__()

    def get_example(self, i):
        if self._profile_get_example:
            result = None
            start = time.time()
            result = self.data.get_example(i)
            end = time.time()
            elapsed_time = (end - start) * 1000 # ms
            
            with open(self._profile_get_example_csv_path, 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow([
                    i, 
                    elapsed_time, 
                    sys.getsizeof(result[0].data) + sys.getsizeof(result[1].data),
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
                ])
            
            return result
        else:
            return self.data.get_example(i)

    class SingleFileData:
        def __init__(self, l, r, inn, outn, file_path):
            self.l = int(l)
            self.r = int(r)
            self.inn = int(inn)
            self.outn = int(outn)
            self.data = np.load(file_path, mmap_mode='r')
            self.data = np.transpose(self.data, (1, 0, 2, 3))

        def __len__(self):
            return self.r - self.l

        def get_example(self, i):
            ind = self.l + i
            return self.data[:self.inn, ind, :, :].astype(np.float32), \
                   self.data[self.inn:self.inn + self.outn, ind, :, :].astype(np.float32)

    class MultipleFileData:
        def __init__(self, l, r, inn, outn, file_paths):
            self.l = int(l)
            self.r = int(r)
            self.inn = int(inn)
            self.outn = int(outn)
            self.file_size = len(file_paths)
            self.files = {}
            self.file_map = {}

            file_index = 0
            offset_index = 0
            for file_path in file_paths:
                file = np.load(file_path, mmap_mode='r')
                file = np.transpose(file, (1, 0, 2, 3))

                file_len = file.shape[1]
                k = list(range(offset_index, offset_index + file_len))
                v = [file_index] * file_len
                self.file_map = {**self.file_map, **dict(zip(k, v))}
                self.files[file_index] = (file, offset_index)

                offset_index += file_len
                file_index += 1

        def __len__(self):
            return self.r - self.l

        def get_example(self, i):
            file = self.files[self.file_map[self.l + i]]
            # adjust index
            ind = self.l + i - file[1]
            return file[0][:self.inn, ind, :, :].astype(np.float32), \
                   file[0][self.inn:self.inn + self.outn, ind, :, :].astype(np.float32)
