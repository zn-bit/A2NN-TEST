import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as trF
from data_process import datasets_catalog as dc


class RandomRotation(object):
    def __init__(self, angle, rotate_shear):
        self.rotate_shear = rotate_shear
        self.angle = angle

    def __call__(self, img):
        n = np.random.randint(0, 4)
        shear = np.random.randint(-self.rotate_shear, self.rotate_shear + 1)
        angle = int(self.angle * n + shear)
        return trF.rotate(img, angle, False, False, None)


def Augmentation(cfg, datasets_name, mode):
    # data augmentation
    _mean = dc.get_mean(datasets_name)
    _std = dc.get_std(datasets_name)
    aug = []

    aug.append(transforms.Resize([cfg.AUG.RESIZE_SIZE, cfg.AUG.RESIZE_SIZE]))
    if mode == 'train':
        # if cfg.AUG.RandomHorizontalFlip:
        aug.append(transforms.RandomHorizontalFlip())  # horizontal flip
        # if cfg.AUG.RandomVerticalFlip:
        #     aug.append(transforms.RandomVerticalFlip())
        # if cfg.AUG.RandomRotation:
        #     aug.append(RandomRotation(angle=90, rotate_shear=0))
        aug.append(transforms.RandomCrop([cfg.AUG.CROP_SIZE, cfg.AUG.CROP_SIZE]))
    else:
        aug.append(transforms.CenterCrop([cfg.AUG.CROP_SIZE, cfg.AUG.CROP_SIZE]))  # center crop
    aug.append(transforms.ToTensor())
    aug.append(transforms.Normalize(mean=_mean, std=_std))
    return transforms.Compose(aug)