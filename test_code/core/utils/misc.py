import os
import torch
import shutil
import logging
import numpy as np
from core.utils.load_model import *


def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l


class IterLRScheduler(object):
    def __init__(self, optimizer, milestones, lr_mults, latest_iter=-1):
        assert len(milestones) == len(lr_mults), "{} vs {}".format(milestones, lr_mults)
        self.milestones = milestones
        self.lr_mults = lr_mults
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        for i, group in enumerate(optimizer.param_groups):
            if 'lr' not in group:
                raise KeyError("param 'lr' is not specified "
                               "in param_groups[{}] when resuming an optimizer".format(i))
        self.latest_iter = latest_iter

    def _get_lr(self):
        try:
            pos = self.milestones.index(self.latest_iter)
        except ValueError:
            return list(map(lambda group: group['lr'], self.optimizer.param_groups))
        except:
            raise Exception('wtf?')
        return list(map(lambda group: group['lr'] * self.lr_mults[pos], self.optimizer.param_groups))

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.latest_iter + 1
        self.latest_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_lr()):
            param_group['lr'] = lr


class EpochLRScheduler(object):
    def __init__(self, optimizer, milestones, lr_mults, latest_epoch=-1):
        assert len(milestones) == len(lr_mults), "{} vs {}".format(milestones, lr_mults)
        self.milestones = milestones
        self.lr_mults = lr_mults
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        for i, group in enumerate(optimizer.param_groups):
            if 'lr' not in group:
                raise KeyError("param 'lr' is not specified "
                               "in param_groups[{}] when resuming an optimizer".format(i))
        self.latest_epoch = latest_epoch

    def _get_lr(self):
        try:
            pos = self.milestones.index(self.latest_epoch)
        except ValueError:
            return list(map(lambda group: group['lr'], self.optimizer.param_groups))
        except:
            raise Exception('wtf?')
        return list(map(lambda group: group['lr'] * self.lr_mults[pos], self.optimizer.param_groups))

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_epoch=None):
        if this_epoch is None:
            this_epoch = self.latest_epoch + 1
        self.latest_epoch = this_epoch
        for param_group, lr in zip(self.optimizer.param_groups, self._get_lr()):
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):  # Return the correct scale.
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    target_all = target.view(1, -1).expand_as(pred)
    # all
    correct = pred.eq(target_all)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []

        else:
            self.count = 0
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.all = 0.0

    def update(self, val):
        if self.length > 0:
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
            self.sum = np.sum(self.history)
        else:
            self.val = val
            self.sum += val
            self.count += 1
            self.avg = self.sum / self.count
            self.all = self.sum


def accuracy_num(output, target, topk=(1,)):  # Return the number of correct values.
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    target_all = target.view(1, -1).expand_as(pred)
    # all
    correct = pred.eq(target_all)

    # res = []
    correct_k = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0)
        # res.append(correct_k.mul_(100.0 / batch_size))
        correct_k.append(correct[:k].contiguous().view(-1).float().sum(0))
    return correct_k


class AverageMeter_num(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []

        else:
            self.count = 0
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.all = 0.0

    def update(self, val, sizes):
        if self.length > 0:  # Not modified yet.
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
            self.sum = np.sum(self.history)
        else:
            self.avg = val / sizes


# def read_dict_txt(conf_path):
#     f = open(conf_path, "r")
#     test_config = eval(f.read())
#     f.close()
#     return test_config


def save_state(state, save_path, best_save):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    latest_path = '{}/checkpoints/latest_checkpoint.pth.tar'.format(save_path)
    best_path = '{}/checkpoints/best_checkpoint.pth.tar'.format(save_path)
    torch.save(state, latest_path)
    # shutil.copyfile(model_path, latest_path)
    if best_save:
        shutil.copyfile(latest_path, best_path)


def save_add_loss_state(state, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = '{}/iter_{}_{}_checkpoint.pth.tar'.format(save_path, state['step'], state['add_loss_name'])
    latest_path = '{}/latest_{}_checkpoint.pth.tar'.format(save_path, state['add_loss_name'])
    torch.save(state, model_path)
    shutil.copyfile(model_path, latest_path)


def load_add_loss_state(path, model):
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        print("=> load from checkpoint '{}'".format(path))
    else:
        assert True, "=> no checkpoint found at '{}'".format(path)
    model.load_state_dict(checkpoint['state_dict'])


def load_state(model_name, path, model, latest_flag=True, optimizer=None, device=None):
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=map_func)
        if 'state_dict' not in checkpoint and 'step' not in checkpoint:
            checkpoint = {'state_dict': checkpoint, 'step': -1}
    elif latest_flag and not os.path.isfile(path):
        latest_file = '{}/checkpoints/latest_checkpoint.pth.tar'.format(path)
        checkpoint = torch.load(latest_file, map_location=map_func)
    else:
        assert True, "=> no checkpoint found at '{}'".format(path)

    if 'dense' in model_name:
        checkpoint['state_dict'] = dense_transfer(checkpoint['state_dict'])

    elif 'res' in model_name:
        checkpoint['state_dict'] = res_transfer(checkpoint['state_dict'])

    ckpt_keys = set(checkpoint['state_dict'].keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    for k in missing_keys:
        print('caution: missing keys from checkpoint {}: {}'.format(path, k))

    copying_layers = {}
    ignoring_layers = {}
    for key in own_keys:
        if key not in ckpt_keys:
            print('caution: add layer: {} is random initialized.'.format(key))
        else:
            if checkpoint['state_dict'][key].shape == model.state_dict()[key].shape:
                copying_layers[key] = checkpoint['state_dict'][key]
            else:
                ignoring_layers[key] = checkpoint['state_dict'][key]
                print('caution: shape mismatched keys from checkpoint {}: {}'.format(path, key))

    model.load_state_dict(copying_layers, strict=True)
    current_step = checkpoint['step']
    current_epoch = checkpoint['epoch']
    current_lr = next(iter(checkpoint['optimizer']['param_groups']))['lr']
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    if optimizer != None and device != None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("=> also loaded optimizer from checkpoint '{}' (iter {})".format(path, current_step))
    return current_epoch, current_step


def load_imgnet_models(model_name, model, path):
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        state_dict = torch.load(path, map_location=map_func)
        if 'res' in model_name:
            state_dict = res_transfer(state_dict)
        elif 'dfl_vgg16' in model_name:
            state_dict = dfl_transfer(state_dict)
        elif 'identity_vgg' in model_name:
            state_dict = identity_transfer(state_dict)
        elif 'sf_vgg' in model_name:
            state_dict = sf_transfer(state_dict)
        elif 'sf_att_vgg16' in model_name:
            state_dict = sf_att_transfer(state_dict)
        elif 'vgg' in model_name:
            state_dict = vgg_transfer(state_dict)

    else:
        assert True, "=> no checkpoint found at '{}'".format(path)

    # if 'fc_.weight' in model.state_dict().keys():
    #     n = model.state_dict()['fc_.weight'].size()[1] / state_dict['fc.weight'].size()[1]
    #     fc_list = []
    #     for i in range(int(n)):
    #         fc_list.append(state_dict['fc.weight'])
    #     state_dict['fc_.weight'] = torch.cat(fc_list, 1)
    #     state_dict['fc_.bias'] = state_dict['fc.bias']

    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    for k in missing_keys:
        print('caution: missing keys from checkpoint {}: {}'.format(path, k))

    model.load_state_dict(state_dict, strict=False)
