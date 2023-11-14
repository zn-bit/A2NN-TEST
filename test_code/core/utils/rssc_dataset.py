import os
import yaml
import random
from easydict import EasyDict

import torch
from PIL import Image
from torch.utils.data import Dataset
from data_process import datasets_catalog as dc
from core.utils.ImgAugmentations import Augmentation


class RSSCDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        self.datasets_name = cfg.TRAIN.DATASETS
        self.num_classes = dc.get_num_classes(self.datasets_name)

        self.mode = mode
        if mode == 'train':
            self.batch_size = cfg.TRAIN.BATCH_SIZE
        else:
            self.batch_size = cfg.TEST.BATCH_SIZE

        self.train_ratio = cfg.TRAIN.TRAIN_RATIO
        self.test_ratio = cfg.TEST.TEST_RATIO
        assert dc.contains(self.datasets_name), 'Unknown dataset_name: {}'.format(self.datasets_name)
        self.transform = Augmentation(cfg, self.datasets_name, self.mode)
        print(self.transform)
        self.metas = []

        source_file = dc.get_source_index(self.datasets_name)[self.mode]
        if self.mode == 'train':
            source_file = source_file.format(self.train_ratio)
        elif self.mode == 'test':
            source_file = source_file.format(self.test_ratio)
        print(source_file)
        prefix = dc.get_prefix(self.datasets_name)
        with open(source_file, 'r') as f:
            lines = f.readlines()
            # random.shuffle(lines)
            for line in lines:
                path = os.path.join(prefix, line.split()[0])
                label = int(line.split()[1])
                self.metas.append((path, label))
        self.num = len(lines)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename = self.metas[idx][0]
        cls = self.metas[idx][1]
        img = Image.open(filename)
        # transform
        if self.transform is not None:
            img = self.transform(img)
        return img, cls


def get_class_counts_dict(name):
    class_counts_dict = {}
    class_file = dc.get_source_index(name)['class_txt']
    with open(class_file, 'r') as class_f:
        lines = class_f.readlines()
        for line in lines:
            img_cls = line.split()[1]
            img_num = float(line.split()[2])
            class_counts_dict[img_cls] = img_num
    return class_counts_dict


def get_difficulty_tensor(name):
    difficulty_list = []
    class_file = dc.get_source_index(name)['class_txt']
    with open(class_file, 'r') as class_f:
        lines = class_f.readlines()
        for line in lines:
            difficulty = float(line.split()[3])
            difficulty_list.append(difficulty)
    difficulty_tensor = torch.tensor(difficulty_list).float()
    return difficulty_tensor


def get_weights_tensor(name, class_counts_dict):
    weights_list = []

    for img_cls in class_counts_dict.keys():
        weight = 1 / float(class_counts_dict[img_cls])
        weights_list.append(weight)
    weights_tensor = torch.tensor(weights_list).float()
    return weights_tensor

# if __name__ == '__main__':
#     cfg_dir = '/home/Project/my_proj/plane/classify_air_2019/cfgs/resnet101_aug_input.yaml'
#     with open(cfg_dir, 'r') as f:
#         config = yaml.load(f)
#         cfg = EasyDict(config)
#
#     train_dataset = RSSCDataset(cfg, mode='train')
#     for idx in range(600):
#         img, cls, filename = train_dataset.__getitem__(idx)
#         tensor_imshow(img)