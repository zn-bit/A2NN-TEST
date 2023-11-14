from __future__ import print_function, division
import torch
import torch.nn as nn
import argparse
from collections import OrderedDict
import yaml
from easydict import EasyDict
from core.utils.sampler_utils import set_seed_, classifier_collate
from core.utils.misc import AverageMeter_num, accuracy_num, create_logger, AverageMeter
from core.utils.rssc_dataset import RSSCDataset
import core.models as models
import data_process.datasets_catalog as dc
from core.utils.compute import AverageMeter, compute_singlecrop

# ============================ yaml调用 ============================
parser = argparse.ArgumentParser(description="test")
parser.add_argument('--config', default='cfgs/UC.yaml')
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    cfg = EasyDict(config)
set_seed_(cfg.TRAIN.SEED)  # 设置随机种子
gpu_ids = cfg.MODEL.DEVICE_IDS
torch.cuda.set_device(gpu_ids[0])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def test_CNN():
    top1_error = AverageMeter()
    top5_error = AverageMeter()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            output = teacher_CNN(images)
            loss = torch.ones(1)
            single_error, single_loss, single5_error = compute_singlecrop(
                outputs=output, loss=loss,
                labels=labels, top5_flag=True, mean_flag=True)
            top1_error.update(single_error, images.size(0))
            top5_error.update(single5_error, images.size(0))
    return top1_error.avg, top5_error.avg


def test_A2NN():
    top1_error = AverageMeter()
    top5_error = AverageMeter()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            output,_ = teacher_AANN(images)
            loss = torch.ones(1)
            single_error, single_loss, single5_error = compute_singlecrop(
                outputs=output, loss=loss,
                labels=labels, top5_flag=True, mean_flag=True)
            top1_error.update(single_error, images.size(0))
            top5_error.update(single5_error, images.size(0))
    return top1_error.avg, top5_error.avg


def test_QA2NN(model,NUM):
    top1_error = AverageMeter()
    top5_error = AverageMeter()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            output,_ = model(images)
            loss = torch.ones(1)
            single_error, single_loss, single5_error = compute_singlecrop(
                outputs=output, loss=loss,
                labels=labels, top5_flag=True, mean_flag=True)
            top1_error.update(single_error, images.size(0))
            top5_error.update(single5_error, images.size(0))
    return top1_error.avg, top5_error.avg


teacher_CNN = models.__dict__[cfg.MODEL.CNN](num_classes=21, cfg=cfg).cuda(gpu_ids[0])
state_dict = teacher_CNN.state_dict()
pretrained_dict = torch.load(cfg.MODEL.CNNDICT, map_location='cuda:' + str(gpu_ids[0]))
new_state_dict = OrderedDict()
for k, v in pretrained_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:]  # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
updated_dict = {k: v for k, v in new_state_dict.items() if k in state_dict}
state_dict.update(updated_dict)
teacher_CNN.load_state_dict(state_dict)
teacher_CNN.eval()


input_size = (1, 3, 224, 224)
num_params = count_parameters(teacher_CNN)
print(f"CNN: Number of parameters: {num_params}")


# ============================ A2NN模型 ============================
teacher_AANN = models.__dict__[cfg.MODEL.A2NN](num_classes=21, cfg=cfg).cuda(gpu_ids[0])
state_dict = teacher_AANN.state_dict()
pretrained_dict = torch.load(cfg.MODEL.A2NNDICT, map_location='cuda:' + str(gpu_ids[0]))
new_state_dict = OrderedDict()
for k, v in pretrained_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:]  # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
updated_dict = {k: v for k, v in new_state_dict.items() if k in state_dict}
state_dict.update(updated_dict)
teacher_AANN.load_state_dict(state_dict)
teacher_AANN.eval()

num_params = count_parameters(teacher_AANN)
print(f"A2NN: Number of parameters: {num_params}")

model = models.__dict__[cfg.MODEL.QA2NN](num_classes=21, cfg=cfg, num_bits_epoch=cfg.MODEL.NUM_BITS).cuda(gpu_ids[0])
state_dict = model.state_dict()
pretrained_dict = torch.load(cfg.MODEL.QA2NNDICT, map_location='cuda:' + str(gpu_ids[0]))
new_state_dict = OrderedDict()
for k, v in pretrained_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:]  # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
updated_dict = {k: v for k, v in new_state_dict.items() if k in state_dict}
state_dict.update(updated_dict)
model.load_state_dict(state_dict)
model.eval()


num_params = count_parameters(model)
print(f"QA2NN: Number of parameters: {num_params}")


# ============================ 数据 ============================
test_dataset = RSSCDataset(cfg, mode='test')
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=test_dataset.batch_size,
    shuffle=False, num_workers=cfg.TRAIN.WORKERS,
    pin_memory=False, collate_fn=classifier_collate)
test_dataset_sizes = test_dataset.num
num_class = dc.get_num_classes(cfg.TRAIN.DATASETS)


# ============================ use gpu or not ============================
use_gpu = torch.cuda.is_available()
print("use_gpu:{}".format(use_gpu))
if use_gpu:
    criterion = nn.CrossEntropyLoss().cuda()

a,b = test_CNN()
c,d = test_A2NN()
e,f = test_QA2NN(model,cfg.MODEL.NUM_BITS)
print(
    "CNN: [top1_acc: %.4f%%] [top5_acc: %.4f%%]"
    % ((100.00 - a), (100.00 - b))
)
print(
    "A2NN: [top1_acc: %.4f%%] [top5_acc: %.4f%%]"
    % ((100.00 - c), (100.00 - d)))
print(
    "QA2NN: [Q_BIT: %d bit] [top1_acc: %.4f%%] [top5_acc: %.4f%%]"
    % (cfg.MODEL.NUM_BITS, (100.00 - e), (100.00 - f)))