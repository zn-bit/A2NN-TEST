import numpy as np
import torch
import random
import os
from torch.utils.data.sampler import Sampler

# ref
def set_seed_(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class TrainIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, last_iter=-1):
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.last_iter = last_iter
        self.total_size = self.total_iter * self.batch_size
        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[(self.last_iter+1)*self.batch_size:])
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self):
        indices = np.arange(len(self.dataset))
        indices = indices[:self.total_size]
        num_repeat = (self.total_size-1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:self.total_size]

        np.random.shuffle(indices)

        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        return self.total_size - (self.last_iter+1)*self.batch_size


class TrainIterationIdSampler(Sampler):
    def __init__(self, dataset, total_iter, id_batch_size, last_iter=-1):
        self.dataset = dataset
        self.total_iter = total_iter
        self.id_batch_size = id_batch_size
        self.last_iter = last_iter
        self.total_id_size = self.total_iter * self.id_batch_size
        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[(self.last_iter+1)*self.id_batch_size:])
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self):
        indices = np.arange(self.dataset.num_classes)
        indices = indices[:self.total_id_size]
        num_repeat = (self.total_id_size-1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:self.total_id_size]

        np.random.shuffle(indices)

        assert len(indices) == self.total_id_size

        return indices

    def __len__(self):
        return self.total_id_size - (self.last_iter+1)*self.id_batch_size


def classifier_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    labels = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        labels.append(sample[1])

    return torch.stack(imgs, 0), torch.LongTensor(labels)


def classifier_collate_test(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    imgs = []
    file_paths = []
    for sample in batch:
        imgs.append(sample[0])
        file_paths.append(sample[1])

    return torch.stack(imgs, 0), file_paths