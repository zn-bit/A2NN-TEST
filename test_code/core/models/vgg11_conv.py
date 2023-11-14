import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['vgg11_conv']


class vgg11_conv(nn.Module):
    def __init__(self, cfg, num_classes=1000, init_weights=True):
        super(vgg11_conv, self).__init__()
        # VGG16 (using return_indices=True on the MaxPool2d layers)
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),

            # conv2
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),

            # conv3
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),

            # conv4
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),

            # conv5
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
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
            nn.Conv2d(512, num_classes, 1, padding=0, bias=False),
            nn.BatchNorm2d(num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        output = self.features(x)
        output = F.adaptive_max_pool2d(output, (1, 1))
        # output = output.view(output.size()[0], -1)
        # fc1 = self.classifier0(output)
        # fc2 = self.classifier1(fc1)
        # fc3 = self.classifier2(fc2)
        output = self.last_linear(output)
        output = output.view(output.size()[0], -1)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)