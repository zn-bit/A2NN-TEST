# 全CONV
# 全部BN
# 全去BIAS
# 全部池化之后激活

import torch
from torch import nn
import torch.nn.functional as F
import adder

__all__ = ['vgg11_adder']


class vgg11_adder(nn.Module):
    def __init__(self, cfg, num_classes=1000, init_weights=True):
        super(vgg11_adder, self).__init__()
        # VGG16 (using return_indices=True on the MaxPool2d layers)
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # conv1
            adder.adder2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),

            # conv2
            adder.adder2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),

            # conv3
            adder.adder2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            adder.adder2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),

            # conv4
            adder.adder2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            adder.adder2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),

            # conv5
            adder.adder2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            adder.adder2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU())

        # self.classifier0 = nn.Sequential(
        #     # nn.Linear(512 * 7 * 7, 4096),
        #     nn.Conv2d(512, 4096, 7, padding=0, bias=False),
        #     nn.BatchNorm2d(4096),
        #     nn.ReLU()
        # )
        # self.classifier1 = nn.Sequential(
        #     # nn.Linear(4096, 4096),
        #     nn.Conv2d(4096, 4096, 1, padding=0, bias=False),
        #     nn.BatchNorm2d(4096),
        #     nn.ReLU()
        # )
        # self.classifier2 = nn.Sequential(
        #     # nn.Linear(4096, 1000)
        #     nn.Conv2d(4096, 1000, 1, padding=0, bias=False),
        #     nn.BatchNorm2d(1000),
        #     nn.ReLU()
        # )

        self.last_linear = nn.Sequential(
            # nn.Linear(1000, num_classes, bias=cfg.MODEL.CLASSIFIER_BIAS)
            adder.adder2d(512, num_classes, 1, padding=0, bias=False),
            nn.BatchNorm2d(num_classes)
        )

        # if init_weights:
        #     self._initialize_weights()

    def forward(self, x):
        features = []
        # output = self.features(x)
        for block in self.features.children():
            x = block(x)
            if block.__module__ == 'torch.nn.modules.batchnorm':
                features.append(x)
            # if block
            # features.extend(fs)
        output = F.adaptive_max_pool2d(x, (1, 1))
        # output = output.view(output.size()[0], -1)
        # fc1 = self.classifier0(output)
        # fc2 = self.classifier1(fc1)
        # fc3 = self.classifier2(fc2)
        # for block in self.last_linear.children():
        #     output = block(output)
        #     # if block.__module__ == 'torch.nn.modules.batchnorm':
        #     #     features.append(output)
        output = self.last_linear(output)
        output = output.view(output.size()[0], -1)
        return output, features

    def get_bn_before_relu(self):
        bn1 = self.features[1]
        bn2 = self.features[5]
        bn3 = self.features[9]
        bn4 = self.features[12]
        bn5 = self.features[16]
        bn6 = self.features[19]
        bn7 = self.features[23]
        bn8 = self.features[26]
        bn9 = self.last_linear[1]

        return [bn1, bn2, bn3, bn4, bn5, bn6, bn7, bn8, bn9]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, adder.adder2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # if m.bias is not None:
                #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
