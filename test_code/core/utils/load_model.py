import re
import torch
from collections import OrderedDict


def res_transfer(checkpoint):
    
    mapped_state_dict = OrderedDict()
    for key, value in checkpoint.items():
        mapped_key = key
        mapped_state_dict[mapped_key] = value
        if 'running_var' in key:
            mapped_state_dict[key.replace('running_var', 'num_batches_tracked')] = torch.zeros(1).cuda()
    return mapped_state_dict


def dense_transfer(state_dict):
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict


def vgg_transfer(state_dict):
    mapped_dict = {}
    for key, value in state_dict.items():
        if 'classifier' not in key:
            mapped_dict[key] = state_dict[key]
    mapped_dict['classifier0.0.weight'] = state_dict['classifier.0.weight']
    mapped_dict['classifier0.0.bias'] = state_dict['classifier.0.bias']
    mapped_dict['classifier1.0.weight'] = state_dict['classifier.3.weight']
    mapped_dict['classifier1.0.bias'] = state_dict['classifier.3.bias']
    mapped_dict['classifier2.0.weight'] = state_dict['classifier.6.weight']
    mapped_dict['classifier2.0.bias'] = state_dict['classifier.6.bias']
    return mapped_dict


def sf_transfer(state_dict):
    mapped_dict = {}
    for key, value in state_dict.items():
        if 'classifier' not in key:
            mapped_dict[key] = state_dict[key]
    mapped_dict['FC_Conv1.0.weight'] = state_dict['classifier.0.weight'].contiguous().view(4096, 512, 7, 7)
    mapped_dict['FC_Conv1.0.bias'] = state_dict['classifier.0.bias']
    mapped_dict['FC_Conv2.0.weight'] = state_dict['classifier.3.weight'].contiguous().view(4096, 256, 4, 4)
    mapped_dict['FC_Conv2.0.bias'] = state_dict['classifier.3.bias']
    return mapped_dict


def sf_att_transfer(state_dict):
    mapped_dict = {}
    for key, value in state_dict.items():
        if 'classifier' not in key:
            mapped_dict[key] = state_dict[key]
    mapped_dict['FC_Conv1.0.weight'] = state_dict['classifier.0.weight'].contiguous().view(4096, 512, 7, 7)
    mapped_dict['FC_Conv1.0.bias'] = state_dict['classifier.0.bias']
    mapped_dict['FC_Conv2.0.weight'] = state_dict['classifier.3.weight'].contiguous().view(4096, 256, 4, 4)
    mapped_dict['FC_Conv2.0.bias'] = state_dict['classifier.3.bias']
    return mapped_dict


def identity_transfer(state_dict):
    mapped_dict = {}
    mapped_dict['conv1.0.weight'] = state_dict['features.0.weight']
    mapped_dict['conv1.0.bias'] = state_dict['features.0.bias']

    mapped_dict['layer1.0.weight'] = state_dict['features.2.weight']
    mapped_dict['layer1.0.bias'] = state_dict['features.2.bias']

    mapped_dict['layer2.0.weight'] = state_dict['features.5.weight']
    mapped_dict['layer2.0.bias'] = state_dict['features.5.bias']
    mapped_dict['layer2.2.weight'] = state_dict['features.7.weight']
    mapped_dict['layer2.2.bias'] = state_dict['features.7.bias']

    mapped_dict['conv2.0.weight'] = mapped_dict['layer2.0.weight']
    mapped_dict['conv2.0.bias'] = mapped_dict['layer2.0.bias']

    mapped_dict['layer3.0.weight'] = state_dict['features.10.weight']
    mapped_dict['layer3.0.bias'] = state_dict['features.10.bias']
    mapped_dict['layer3.2.weight'] = state_dict['features.12.weight']
    mapped_dict['layer3.2.bias'] = state_dict['features.12.bias']
    mapped_dict['layer3.4.weight'] = state_dict['features.14.weight']
    mapped_dict['layer3.4.bias'] = state_dict['features.14.bias']

    mapped_dict['conv3.0.weight'] = mapped_dict['layer3.0.weight']
    mapped_dict['conv3.0.bias'] = mapped_dict['layer3.0.bias']

    mapped_dict['layer4.0.weight'] = state_dict['features.17.weight']
    mapped_dict['layer4.0.bias'] = state_dict['features.17.bias']
    mapped_dict['layer4.2.weight'] = state_dict['features.19.weight']
    mapped_dict['layer4.2.bias'] = state_dict['features.19.bias']
    mapped_dict['layer4.4.weight'] = state_dict['features.21.weight']
    mapped_dict['layer4.4.bias'] = state_dict['features.21.bias']

    mapped_dict['conv4.0.weight'] = mapped_dict['layer4.0.weight']
    mapped_dict['conv4.0.bias'] = mapped_dict['layer4.0.bias']

    mapped_dict['layer5.0.weight'] = state_dict['features.24.weight']
    mapped_dict['layer5.0.bias'] = state_dict['features.24.bias']
    mapped_dict['layer5.2.weight'] = state_dict['features.26.weight']
    mapped_dict['layer5.2.bias'] = state_dict['features.26.bias']
    mapped_dict['layer5.4.weight'] = state_dict['features.28.weight']
    mapped_dict['layer5.4.bias'] = state_dict['features.28.bias']

    mapped_dict['FC_Conv1.0.weight'] = state_dict['classifier.0.weight'].contiguous().view(4096, 512, 7, 7)
    mapped_dict['FC_Conv1.0.bias'] = state_dict['classifier.0.bias']
    mapped_dict['FC_Conv2.0.weight'] = state_dict['classifier.3.weight'].contiguous().view(4096, 256, 4, 4)
    mapped_dict['FC_Conv2.0.bias'] = state_dict['classifier.3.bias']
    return mapped_dict


def dfl_transfer(state_dict):
    mapped_dict = {}
    mapped_dict['conv1_4.0.weight'] = state_dict['features.0.weight']
    mapped_dict['conv1_4.0.bias'] = state_dict['features.0.bias']
    mapped_dict['conv1_4.2.weight'] = state_dict['features.2.weight']
    mapped_dict['conv1_4.2.bias'] = state_dict['features.2.bias']

    mapped_dict['conv1_4.5.weight'] = state_dict['features.5.weight']
    mapped_dict['conv1_4.5.bias'] = state_dict['features.5.bias']
    mapped_dict['conv1_4.7.weight'] = state_dict['features.7.weight']
    mapped_dict['conv1_4.7.bias'] = state_dict['features.7.bias']

    mapped_dict['conv1_4.10.weight'] = state_dict['features.10.weight']
    mapped_dict['conv1_4.10.bias'] = state_dict['features.10.bias']
    mapped_dict['conv1_4.12.weight'] = state_dict['features.12.weight']
    mapped_dict['conv1_4.12.bias'] = state_dict['features.12.bias']
    mapped_dict['conv1_4.14.weight'] = state_dict['features.14.weight']
    mapped_dict['conv1_4.14.bias'] = state_dict['features.14.bias']

    mapped_dict['conv1_4.17.weight'] = state_dict['features.17.weight']
    mapped_dict['conv1_4.17.bias'] = state_dict['features.17.bias']
    mapped_dict['conv1_4.19.weight'] = state_dict['features.19.weight']
    mapped_dict['conv1_4.19.bias'] = state_dict['features.19.bias']
    mapped_dict['conv1_4.21.weight'] = state_dict['features.21.weight']
    mapped_dict['conv1_4.21.bias'] = state_dict['features.21.bias']

    mapped_dict['conv5.0.weight'] = state_dict['features.24.weight']
    mapped_dict['conv5.0.bias'] = state_dict['features.24.bias']
    mapped_dict['conv5.2.weight'] = state_dict['features.26.weight']
    mapped_dict['conv5.2.bias'] = state_dict['features.26.bias']
    mapped_dict['conv5.4.weight'] = state_dict['features.28.weight']
    mapped_dict['conv5.4.bias'] = state_dict['features.28.bias']

    mapped_dict['conv5.0.weight'] = state_dict['features.24.weight']
    mapped_dict['conv5.0.bias'] = state_dict['features.24.bias']
    mapped_dict['conv5.2.weight'] = state_dict['features.26.weight']
    mapped_dict['conv5.2.bias'] = state_dict['features.26.bias']
    mapped_dict['conv5.4.weight'] = state_dict['features.28.weight']
    mapped_dict['conv5.4.bias'] = state_dict['features.28.bias']
    return mapped_dict