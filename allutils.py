#! -*- coding: utf-8 -*-

from __future__ import print_function, division
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
import shutil
import numpy as np
import time
import  random
import copy
from collections import Counter
import math

dp = copy.deepcopy

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_imagenet(args):
    MEAN_RGB = (0.485, 0.456, 0.406)
    VAR_RGB = (0.229, 0.224, 0.225)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_RGB, VAR_RGB),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_RGB, VAR_RGB),
    ])
    return transform_train, transform_test

def _data_transforms_cifar10(cutout=False, cutout_length=16):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if cutout:
    train_transform.transforms.append(Cutout(cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def get_dataloader_cifar10(root, batch_size, nw, dataset_type='train', cutout=False, cutout_length=16):
    train_transform, valid_transform = _data_transforms_cifar10(cutout=cutout, cutout_length=cutout_length)
    if dataset_type == 'train':
        train_data = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                  num_workers=nw)
        print('Succeed to init train cifar10 train DataLoader!')
        return trainloader
    elif dataset_type == 'val' or dataset_type == 'valid':
        valid_data = datasets.CIFAR10(root=root, train=False, download=True, transform=valid_transform)
        validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False,
                                                  pin_memory=True, num_workers=nw)
        print('Succeed to init valid cifar10 val DataLoader!')
        return validloader
    else:
        raise Exception('cifar10 DataLoader: Unknown dataset type -- %s' % dataset_type)

def get_imagenet_dataloader(args, batch_size, dataset_root='../../imagenet12/', dataset_type='train', nw=8):
    TRAIN_DIR = 'train'
    VALIDATION_DIR = 'val'
    TEST_DIR = 'test'
    PARTIAL_DIR = 'partial'
    # PARTIAL_DIR = 'val'
    transform_train, transform_test = _data_transforms_imagenet(args)
    if dataset_type == 'train':
        train_dataset_root = os.path.join(dataset_root, TRAIN_DIR)
        trainset = datasets.ImageFolder(root=train_dataset_root, transform=transform_train)
        trainloader = DataLoader(trainset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=nw,
                                 pin_memory=True,
                                 drop_last=False)
        print('Succeed to init train ImageNet train DataLoader!')
        return trainloader
    elif dataset_type == 'partial':
        partial_dataset_root = os.path.join(dataset_root, PARTIAL_DIR)
        partialset = datasets.ImageFolder(root=partial_dataset_root, transform=transform_train)
        partialloader = DataLoader(partialset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=nw,
                                 pin_memory=True,
                                 drop_last=False)
        print('Succeed to init partial ImageNet train DataLoader!')
        return partialloader
    elif dataset_type == 'val' or dataset_type == 'valid':
        val_dataset_root = os.path.join(dataset_root, VALIDATION_DIR)
        valset = datasets.ImageFolder(root=val_dataset_root, transform=transform_test)
        valloader = DataLoader(valset,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=nw,
                               pin_memory=True,
                               drop_last=False)
        print('Succeed to init val ImageNet val DataLoader!')
        return valloader
    elif dataset_type == 'test':
        test_dataset_root = os.path.join(dataset_root, TEST_DIR)
        testset = datasets.ImageFolder(root=test_dataset_root, transform=transform_test)
        testloader = DataLoader(testset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=nw,
                                pin_memory=True,
                                drop_last=False)
        print('Succeed to init test ImageNet test DataLoader!')
        return testloader
    else:
        raise Exception('IMAGENET DataLoader: Unknown dataset type -- %s' % dataset_type)

def accuracy(output, target, topk=(1,)):
    """
    Calc top1 and top5
    :param output: logits
    :param target: groundtruth
    :param topk: top1 and top5
    :return:
    """
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0))
    return res

def save_all(model, optimizer, scheduler, args, epoch, dir, parallel):
    if parallel:
        torch.save({
            "model":model,
            "module":model.module,
            "optim":optimizer,
            "scheduler":scheduler,
            "args":args,
            "epoch":epoch,
            "modelstate":model.module.state_dict()
        }, dir)
    else:
        torch.save({
            "model": model,
            "optim": optimizer,
            "scheduler": scheduler,
            "args": args,
            "epoch": epoch,
            "modelstate": model.state_dict()
        }, dir)

def save_checkpoint(model, optimizer, epoch, dir, parallel):
    if parallel:
        torch.save({"model": model.module.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch}, dir)
    else:
        torch.save({"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch}, dir)


def load_checkpoint(model, optimizer, scheduler, dir, parallel):
    checkpoint = torch.load(dir, map_location=torch.device('cpu'))
    # model_statedict = move2cpu(checkpoint, parallel)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optim"])
    if scheduler is not None:
        scheduler.last_epoch = checkpoint["epoch"]


def move2cpu(checkpoint, parallel):
    model_statedict = checkpoint['model']
    if parallel:
        keys = list(model_statedict.keys())
        for key in keys:
            new_key = key[7:]  # "module." 7涓瓧绗?
            model_statedict[new_key] = model_statedict.pop(key)
    return model_statedict

def load_model_statedict(model_dir, epoch_id, parallel):
    model_path = os.path.join(model_dir,'checkpoints', "epoch_{}".format(epoch_id))
    assert os.path.exists(model_path), "model weights not exist"
    model_statedict_path = os.path.join(model_dir, "statedict_epoch_{}".format(epoch_id))
    if os.path.exists(model_statedict_path):
        model_statedict = torch.load(model_statedict_path)
    else:
        if parallel:
            model_statedict = torch.load(model_path)['model']
            keys = list(model_statedict.keys())
            for key in keys:
                new_key = key[7:]  # "module." 7涓瓧绗?
                model_statedict[new_key] = model_statedict.pop(key)
        torch.save(model_statedict, model_statedict_path)
    return model_statedict


def save_total_model(model, model_path, parallel_able):
    """
    将模型参数保存到cpu中，下次恢复时直接恢复就好，使用cuda()载入到gpu中
    :param model:
    :param model_path: 模型保存路径
    :param parallel_able: 训练时是否使用了多个gpu并行训练
    :return:
    """
    if parallel_able:
        torch.save(model.module, model_path)
    else:
        torch.save(model, model_path)


def save(model, model_path, parallel_able):
    """
    将模型参数保存到cpu中，下次恢复时直接恢复就好，使用cuda()载入到gpu中
    :param model:
    :param model_path: 模型保存路径
    :param parallel_able: 训练时是否使用了多个gpu并行训练
    :return:
    """
    if parallel_able:
        state = model.module.state_dict()
    else:
        state = model.state_dict()
    for key in state:
        state[key] = state[key].clone().cpu()
    torch.save(state, model_path)
    return state


def load(model, model_path, strict=False):
    """
    加载模型参数，模型在cpu中。如果需要载入gpu中，需要在函数外部使用model.cuda()
    :param model:
    :param model_path:
    :param strict:
    :return:
    """
    model.load_state_dict(torch.load(model_path), strict=strict)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(path + "/checkpoints")
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


class SmoothingCrossEntropyLoss(torch.nn.Module):

    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        log_prob = nn.functional.log_softmax(inputs, dim=1)
        targets = torch.zeros_like(log_prob).scatter(dim=1, index=targets.unsqueeze(1), value=1)
        targets = (1 - self.smoothing) * targets + self.smoothing / (targets.size(1) - 1) * (1 - targets)
        loss = -torch.sum(targets * log_prob, dim=1).mean()
        return loss

def train_cifar10(model, epoch, trainloader, optimizer, criterion, writer, grad_clip, frequency = 100):
    model.train()
    # model.set_prob(.5)
    total_counter = 0
    total_loss = 0.0
    total_top1 = 0.0
    dataset_nums = len(trainloader.dataset)
    st = time.time()
    for iters, (images, labels) in enumerate(trainloader):
        images, labels = images.cuda(), labels.cuda()
        # if (iters + 1) % 50 == 0:
        #     model.random_path()
        # model.random_path()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        top1 = sum((torch.max(outputs, 1)[1] == labels)).item()
        total_counter += images.size(0)
        loss = loss.item()
        total_top1 += top1
        total_loss += loss
        if writer:
            writer.add_scalars('train', {'acc': top1, 'loss': loss}, iters)
        if (iters + 1) % frequency == 0 or total_counter == dataset_nums:
            dt = time.time() - st
            st = time.time()
            mean_top1 = total_top1 / total_counter
            mean_loss = total_loss / total_counter
            process = total_counter / dataset_nums * 100
            logging.info(
                'Epoch: {:3d}\tProcess: {:3.2f}% ({:7d}/{:7d})\tmloss: {:.4f}\tmTop1: {:.4f}\t{}itertime: {:.4f}'.format(
                    epoch, process, total_counter, dataset_nums, mean_loss, mean_top1, frequency, dt))

    return mean_top1, mean_loss

def test_cifar10(model, testloader, writer=None, log=False):
    model.eval()
    # model.set_prob(1.)
    total_counter = 0
    total_top1 = 0.0
    st = time.time()
    with torch.no_grad():
        for iters, (images, labels) in enumerate(testloader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            top1 = sum((torch.max(outputs, 1)[1] == labels)).item()
            total_counter += images.size(0)
            total_top1 += top1
            if writer is not None:
                writer.add_scalar(top1, iters)
    mean_top1 = total_top1 / total_counter
    dt = time.time() - st
    if log:
        logging.info('mTop1: {:.4f}\tcost time: {:.4f}'.format(mean_top1, dt))
    return mean_top1


def test_train_cifar10(model, testloader, trainloader=None, writer=None, log=False):
    with torch.no_grad():
        if trainloader is not None:
            model.train()
            batch_size = trainloader.batch_size
            for iters, (images, labels) in enumerate(trainloader):
                images, labels = images.cuda(), labels.cuda()
                _ = model(images)
                if (iters + 1) * batch_size > 2000:
                    break

        model.eval()
        total_counter = 0
        total_top1 = 0.0
        st = time.time()
        for iters, (images, labels) in enumerate(testloader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            top1 = sum((torch.max(outputs, 1)[1] == labels)).item()
            total_counter += images.size(0)
            total_top1 += top1
            if writer is not None:
                writer.add_scalar(top1, iters)
        mean_top1 = total_top1 / total_counter
        dt = time.time() - st
        if log:
            logging.info('mTop1: {:.4f}\tcost time: {:.4f}'.format(mean_top1, dt))
    return mean_top1


def train(model, epoch, trainloader, optimizer, criterion, frequency=100):
    # model.train()
    total_top1 = 0.0
    # total_top5 = 0.0
    total_counter = 0
    total_loss = 0.0
    dataset_nums = len(trainloader.dataset)
    st = time.time()
    for iter, (images, labels) in enumerate(trainloader):
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        top1 = accuracy(outputs, labels, topk=(1,))[0]
        total_counter += images.size(0)
        total_top1 += top1.item()
        # total_top5 += top5.item()
        total_loss += loss.item()

        if (iter + 1) % frequency == 0:
            dt = time.time() - st
            st = time.time()
            mean_top1 = total_top1 / total_counter
            # mean_top5 = total_top5 / total_counter
            mean_loss = total_loss / total_counter
            process = total_counter / dataset_nums * 100
            # logging.info(
            #     'Epoch: {:3d}\tProcess: {:2.2f}% ({:7d}/{:7d})\tmloss: {:.4f}\tmTop1: {:.4f}\tmTop5: {:.6f}'.format(
            #         epoch, process, total_counter, dataset_nums, mean_loss, mean_top1, mean_top5))
            logging.info(
                'Epoch: {:3d}\tProcess: {:2.2f}% ({:7d}/{:7d})\tmloss: {:.4f}\tmTop1: {:.4f}\t{}itertime: {:.4f}'.format(
                    epoch, process, total_counter, dataset_nums, mean_loss, mean_top1, frequency, dt))
            # print(
            #     'Epoch: {:3d}\tProcess: {:2.2f}% ({:7d}/{:7d})\tmloss: {:.4f}\tmTop1: {:.4f}'.format(
            #         epoch, process, total_counter, dataset_nums, mean_loss, mean_top1))
    mean_top1 = total_top1 / total_counter
    # mean_top5 = total_top5 / total_counter
    mean_loss = total_loss / total_counter
    # logging.info(
    #     'Epoch: {:3d}\tmloss: {:.4f}\tmTop1: {:.4f}'.format(
    #         epoch, mean_loss, mean_top1))

    # return mean_top1

def train_iter(model, epoch, trainloader, optimizer, criterion, frequency = 100, iters = 1000):
    # model.train()
    total_top1 = 0.0
    # total_top5 = 0.0
    total_counter = 0
    total_loss = 0.0
    dataset_nums = len(trainloader.dataset)
    st = time.time()
    for iter, (images, labels) in enumerate(trainloader):
        if iter == iters:
            break
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        top1 = accuracy(outputs, labels, topk=(1,))[0]
        total_counter += images.size(0)
        total_top1 += top1.item()
        # total_top5 += top5.item()
        total_loss += loss.item()

        if (iter + 1) % frequency == 0:
            dt = time.time() - st
            st = time.time()
            mean_top1 = total_top1 / total_counter
            # mean_top5 = total_top5 / total_counter
            mean_loss = total_loss / total_counter
            process = total_counter / dataset_nums * 100
            # logging.info(
            #     'Epoch: {:3d}\tProcess: {:2.2f}% ({:7d}/{:7d})\tmloss: {:.4f}\tmTop1: {:.4f}\tmTop5: {:.6f}'.format(
            #         epoch, process, total_counter, dataset_nums, mean_loss, mean_top1, mean_top5))
            logging.info(
                'Epoch: {:3d}\tProcess: {:2.2f}% ({:7d}/{:7d})\tmloss: {:.4f}\tmTop1: {:.4f}\t{}itertime: {:.4f}'.format(
                    epoch, process, total_counter, dataset_nums, mean_loss, mean_top1, frequency, dt))
            # print(
            #     'Epoch: {:3d}\tProcess: {:2.2f}% ({:7d}/{:7d})\tmloss: {:.4f}\tmTop1: {:.4f}'.format(
            #         epoch, process, total_counter, dataset_nums, mean_loss, mean_top1))
    mean_top1 = total_top1 / total_counter
    # mean_top5 = total_top5 / total_counter
    mean_loss = total_loss / total_counter
    logging.info(
        'Epoch: {:3d}\tmloss: {:.4f}\tmTop1: {:.4f}'.format(
            epoch, mean_loss, mean_top1))

def test(model, testloader, epoch=0, criterion=None):
    model.eval()
    total_top1 = 0.0
    # total_top5 = 0.0
    total_counter = 0.0
    # total_loss = 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            # loss = criterion(outputs, labels)
            # top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            # top1 = accuracy(outputs, labels, topk=(1,))[0]
            total_counter += images.size(0)
            # total_top1 += top1.item()
            total_top1 += sum((torch.max(outputs, 1)[1] == labels)).item()
            # total_top5 += top5.item()
            # total_loss += loss.item()
    mean_top1 = total_top1 / total_counter * 100
    # print(mean_top1)
    # mean_top5 = total_top5 / total_counter
    # mean_loss = total_loss / total_counter
    # logging.info('Epoch: {:3d}\tmTop1: {:.4f}'.format(epoch, mean_top1))
    # print('Epoch: {:3d}\tmTop1: {:.4f}'.format(epoch, mean_top1))
    return mean_top1
    return mean_loss, mean_top1, mean_top5


def train_epoch_iter(model, epoch, trainloader, optimizer, criterion, getpath, params_dict):
    total_top1 = 0.0
    total_top5 = 0.0
    total_counter = 0
    total_loss = 0.0
    dataset_nums = len(trainloader.dataset)
    st100 = time.time()
    st = time.time()
    st1 = time.time()
    total_t = 0.
    for iter, (images, labels) in enumerate(trainloader):
        print("next data", time.time() - st)
        if iter == 0:
            path = getpath()
            update_optim_params(params_dict, path, optimizer)
        st = time.time()

        images, labels = images.cuda(), labels.cuda()
        print("data cuda", time.time() - st)
        st = time.time()
        outputs = model(images, path)
        print("model infer", time.time() - st)
        st = time.time()

        loss = criterion(outputs, labels)
        loss = torch.mean(loss)
        print("loss cal", time.time() - st)
        st = time.time()

        optimizer.zero_grad()
        print("optim zero", time.time() - st)
        st = time.time()

        loss.backward()
        print("loss backward", time.time() - st)

        st = time.time()
        optimizer.step()
        print("optime step", time.time() - st)

        if type(outputs) == list:
            outputs = torch.cat(outputs, 0)
        st = time.time()
        top1, top5 = accuracy(outputs, labels, topk=(1, 5))
        print("acc cal ", time.time() - st)

        total_counter += images.size(0)
        total_top1 += top1.item()
        total_top5 += top5.item()
        total_loss += loss.item()
        print("iter1:", time.time() - st1)
        total_t += time.time() - st1
        if (iter + 1) % 100 == 0:
            et = time.time()
            dt = et - st100
            mean_top1 = total_top1 / total_counter
            mean_top5 = total_top5 / total_counter
            mean_loss = total_loss / total_counter
            process = total_counter / dataset_nums * 100
            logging.info(
                'Epoch: {:3d}\tProcess: {:2.2f}% ({:7d}/{:7d})\tmloss: {:.4f}\tmTop1: {:.4f}\tmTop5: {:.6f}\t100itertime:{:3.2f}'.format(
                    epoch, process, total_counter, dataset_nums, mean_loss, mean_top1, mean_top5, dt))
            # logging.info('Epoch: {:3d}\tProcess: {:2.2f}% ({:7d}/{:7d})\tmloss: {:.4f}\tmTop1: {:.4f}\tmTop5: {:.6f}'.format(epoch, process, total_counter, dataset_nums, mean_loss, mean_top1, mean_top5))
            st100 = time.time()
            print("100 iter time ", total_t)
            total_t = 0.
        st = time.time()
        st1 = time.time()


def get_params(model, parallel_enable):
    params_dict = {}
    if parallel_enable:
        for layer_idx in range(model.module.n_layers):
            params_dict[layer_idx] = {}
            for choice_idx, _ in enumerate(model.module.choices):
                name = "layer{}_choice{}".format(layer_idx, choice_idx)
                params = model.module.mb_flow[name].parameters()
                p_list = []
                for p in params:
                    p_list.append(p)
                params_dict[layer_idx][choice_idx] = p_list
    else:
        for layer_idx in range(model.n_layers):
            params_dict[layer_idx] = {}
            for choice_idx, _ in enumerate(model.choices):
                name = "layer{}_choice{}".format(layer_idx, choice_idx)
                params = model.mb_flow[name].parameters()
                p_list = []
                for p in params:
                    p_list.append(p)
                params_dict[layer_idx][choice_idx] = p_list

    return params_dict


def update_optim_params(params_dict, path, optimizer):
    p_list = []
    for layer_idx, choice in enumerate(path):
        p_list.extend(params_dict[layer_idx][choice])
    optimizer.param_groups[0]['params'] = p_list


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        # print("enter preload")
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            # print("enter stream")

            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            # print("leave stream")
        # print('leave preload')

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


def get_imagenet_dataloader_v2(batch_size, dataset_root='../../imagenet12/', dataset_type='train', nw=4):
    TRAIN_DIR = 'train'
    VALIDATION_DIR = 'val'
    TEST_DIR = 'test'
    PARTIAL_DIR = 'partial'
    # PARTIAL_DIR = 'val'
    if dataset_type == 'train':
        train_dataset_root = os.path.join(dataset_root, TRAIN_DIR)
        train_dataset = datasets.ImageFolder(
            train_dataset_root,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(), Too slow
                # normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=nw, pin_memory=True, collate_fn=fast_collate, drop_last=False)
        print('Succeed to init ImageNet train DataLoader!')
        return trainloader
    elif dataset_type == 'val' or dataset_type == 'valid':
        val_dataset_root = os.path.join(dataset_root, VALIDATION_DIR)
        valset = datasets.ImageFolder(root=val_dataset_root, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]))
        valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=batch_size, shuffle=False,
            num_workers=nw, pin_memory=True,
            collate_fn=fast_collate, drop_last=False)
        print('Succeed to init ImageNet val DataLoader!')
        return valloader
    elif dataset_type == 'test':
        test_dataset_root = os.path.join(dataset_root, TEST_DIR)
        testset = datasets.ImageFolder(root=test_dataset_root, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]))
        testloader = DataLoader(testset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=nw,
                                pin_memory=True,
                                drop_last=False,
                                collate_fn=fast_collate)
        print('Succeed to init ImageNet test DataLoader!')
        return testloader
    else:
        raise Exception('IMAGENET DataLoader: Unknown dataset type -- %s' % dataset_type)



def test_loss(model, epoch, testloader, criterion=None):
    model.eval()
    total_top1 = 0.0
    total_top5 = 0.0
    total_counter = 0.0
    total_loss = 0.0
    prefetcher = data_prefetcher(testloader)
    images, labels = prefetcher.next()
    iter = 0
    while images is not None:
        iter += 1
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        top1, top5 = accuracy(outputs, labels, topk=(1, 5))
        total_counter += images.size(0)
        total_top1 += top1.item()
        total_top5 += top5.item()
        total_loss += loss.item()
        images, labels = prefetcher.next()

    mean_top1 = total_top1 / total_counter
    mean_top5 = total_top5 / total_counter
    mean_loss = total_loss / total_counter
    logging.info(
        'Epoch: {:3d}\tmloss: {:.4f}\tmTop1: {:.4f}\tmTop5:{:.6f}'.format(epoch, mean_loss, mean_top1, mean_top5))
    return mean_loss, mean_top1, mean_top5

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt




############################################################
# special utils
###########################################################

def get_path(layernums, choice_num):
    path = [random.randint(0, choice_num - 1) for _ in range(layernums)]
    first_reduction, second_reduction = layernums // 3 * 1, layernums // 3 * 2
    path[first_reduction] = random.randint(0, choice_num - 2)
    path[second_reduction] = random.randint(0, choice_num - 2)
    return path

def combine_path(path1, path2):
    path = []
    for i in range(len(path1)):
        tmp = set([path1[i], path2[i]])
        path.append(tmp)
    return path


def test(model, two_genes, validloader):
    model.eval()
    two_accs = [0, 0]
    n = 0
    with torch.no_grad():
        for i in range(4):
            images, labels = next(iter(validloader))
            images, labels = images.cuda(), labels.cuda()
            model.set_gene(two_genes[0])
            outputs = model(images)
            top1 = sum((torch.max(outputs, 1)[1] == labels)).item()
            two_accs[0] += top1
            model.set_gene(two_genes[1])
            outputs = model(images)
            top1 = sum((torch.max(outputs, 1)[1] == labels)).item()
            two_accs[1] += top1
            n += labels.size(0)
    model.train()
    return [v / n for v in two_accs]


def ea_process_first(model, trainloader, validloader, criterion, optimizer, scheduler, total_iter, train_iter, pop, save_path):
    history = [[], [], [], [], [], [], []]
    gene_history = []
    diversity_history = []
    domination_history = []
    winners_history = []
    st = time.time()
    tt = 0
    for iteration in range(total_iter):
        model.train()
        two_gene_ids, two_genes = pop.get_two_gene()
        rank = [0, 1]

        # one shot train
        for iters in range(train_iter):
            random.shuffle(rank)
            for i in rank:
                model.set_gene(two_genes[i])
                images, labels = next(iter(trainloader))
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        two_accs = test(model, two_genes, validloader)
        selected_id = two_accs.index(max(two_accs))
        unselected_id = 1 - selected_id
        selected = two_gene_ids[selected_id]
        unselected = two_gene_ids[unselected_id]
        pop.update(unselected=unselected, selected=selected, large_mutate=True, combine=False)
        logging.info('iter:{:4d}/{}\ttrain acc:{}'.format(iteration, total_iter,  two_accs[selected_id]))
        scheduler.step()
        winners = pop.get_winners()
        winners_history.append(winners)
        gene_count = getcountpath(winners)
        gene_pool = trans_pop(winners)
        ent = cal_ent_pool(gene_pool)
        history[0].append(iteration)    # iteration
        history[1].append(gene_count)   # gene count at i_th iteration
        history[2].append(ent)          # gene pool's ent at i_th iteration
        history[3].append(winners)      # all winners at i_th iteration
        history[4].append(two_accs[selected_id])
        if (iteration + 1) % 10 == 0:
            tt += time.time() - st
            total_acc = 0
            for gene in winners:
                model.set_gene(gene)
                acc = test_cifar10(model, validloader)
                total_acc += acc
            avg_acc = total_acc / len(winners)
            model.set_gene(gene_count)
            count_acc = test_cifar10(model, validloader)
            history[-2].append(avg_acc)
            history[-1].append(count_acc)
            st =time.time()
    tt += time.time() - st
    torch.save(history, os.path.join(save_path, 'history'))
    min_ent_idx = history[2][total_iter*4//5:].index(min(history[2][total_iter*4//5:]))
    min_ent_gene = history[1][total_iter*4//5:][min_ent_idx]
    finale_iter_gene = history[1][-1]
    logging.info("min_ent_idx:{}\tmin_ent_gene:\n{}\nfinale_iter_gene:\n{}".format(min_ent_idx+total_iter*4//5, min_ent_gene, finale_iter_gene))
    torch.save(min_ent_gene, os.path.join(save_path, 'min_ent_gene'))
    print('search over')

    return history
   

def oneshot_train2(model, trainloader, validloader, criterion, optimizer, get_random_gene, epoch):
    model.train()
    total_counter = 0
    total_loss = 0.0
    total_top1 = 0.0
    dataset_nums = len(trainloader.dataset)
    st = time.time()
    for iters, (images, labels) in enumerate(trainloader):
        images, labels = images.cuda(), labels.cuda()
        if (iters + 1) % 1 == 0:
        # if iters==0:
            gene = get_random_gene([2, 3, 4, 5], 7)
            model.set_gene(gene)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        top1 = sum((torch.max(outputs, 1)[1] == labels)).item()
        total_counter += images.size(0)
        loss = loss.item()
        total_top1 += top1
        total_loss += loss
        if (iters + 1) % 100 == 0 or total_counter == dataset_nums:
            dt = time.time() - st
            st = time.time()
            mean_top1 = total_top1 / total_counter
            mean_loss = total_loss / total_counter
            process = total_counter / dataset_nums * 100
            logging.info(
                'Epoch: {:3d}\tProcess: {:3.2f}% ({:7d}/{:7d})\tmloss: {:.4f}\tmTop1: {:.4f}\t{}itertime: {:.4f}'.format(
                    epoch, process, total_counter, dataset_nums, mean_loss, mean_top1, 100, dt))

    return mean_top1, mean_loss

def oneshot_train1(model, trainloader, validloader, criterion, optimizer, epoch, pop):
    model.train()
    total_counter = 0
    total_loss = 0.0
    total_top1 = 0.0
    dataset_nums = len(trainloader.dataset)
    st = time.time()
    for iters, (images, labels) in enumerate(trainloader):
        images, labels = images.cuda(), labels.cuda()
        if (iters + 1) % 1 == 0:
            gene = pop.random_identity()
            model.set_gene(gene)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        top1 = sum((torch.max(outputs, 1)[1] == labels)).item()
        total_counter += images.size(0)
        loss = loss.item()
        total_top1 += top1
        total_loss += loss
        if (iters + 1) % 100 == 0 or total_counter == dataset_nums:
            dt = time.time() - st
            st = time.time()
            mean_top1 = total_top1 / total_counter
            mean_loss = total_loss / total_counter
            process = total_counter / dataset_nums * 100
            logging.info(
                'Epoch: {:3d}\tProcess: {:3.2f}% ({:7d}/{:7d})\tmloss: {:.4f}\tmTop1: {:.4f}\t{}itertime: {:.4f}'.format(
                    epoch, process, total_counter, dataset_nums, mean_loss, mean_top1, 100, dt))

    return mean_top1, mean_loss

def train_finale(model, trainloader, criterion, optimizer, epoch, auxiliary, auxiliary_weight, grad_clip):
    model.train()
    total_counter = 0
    total_loss = 0.0
    total_top1 = 0.0
    dataset_nums = len(trainloader.dataset)
    st = time.time()
    for iters, (images, labels) in enumerate(trainloader):
        images, labels = images.cuda(), labels.cuda()
        outputs, aux_outputs = model(images)
        loss = criterion(outputs, labels)
        if auxiliary:
            loss_aux = criterion(aux_outputs, labels)
            loss += auxiliary_weight * loss_aux
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        top1 = sum((torch.max(outputs, 1)[1] == labels)).item()
        total_counter += images.size(0)
        loss = loss.item()
        total_top1 += top1
        total_loss += loss
        if (iters + 1) % 100 == 0 or total_counter == dataset_nums:
            dt = time.time() - st
            st = time.time()
            mean_top1 = total_top1 / total_counter
            mean_loss = total_loss / total_counter
            process = total_counter / dataset_nums * 100
            logging.info(
                'Epoch: {:3d}\tProcess: {:3.2f}% ({:7d}/{:7d})\tmloss: {:.4f}\tmTop1: {:.4f}\t{}itertime: {:.4f}'.format(
                    epoch, process, total_counter, dataset_nums, mean_loss, mean_top1, 100, dt))


    return mean_top1, mean_loss

def test_finale(model, validloader):
    model.eval()
    # model.set_prob(1.)
    total_counter = 0
    total_top1 = 0.0
    with torch.no_grad():
        for iters, (images, labels) in enumerate(validloader):
            images, labels = images.cuda(), labels.cuda()
            outputs, _ = model(images)
            top1 = sum((torch.max(outputs, 1)[1] == labels)).item()
            total_counter += images.size(0)
            total_top1 += top1
    mean_top1 = total_top1 / total_counter
    return mean_top1

def getcountpath(winners):
    pop = copy.deepcopy(winners)
    ele = []
    for gene in pop:
        tmp = []
        for cell in gene:
            for node in cell:
                for element in node:
                    tmp.append(element)
        ele.append(tmp)
    ans = np.array(ele[0])*0
    for i in range(len(ele[0])):
        tmp0 = {k:0 for k in range(7)}
        tmp1 = {k:0 for k in range(7)}
        for j in range(len(ele)):
            tmp0[ele[j][i][0]] += 1
            tmp1[ele[j][i][1]] += 1
        ans[i][0] = max(tmp0, key=tmp0.get)
        ans[i][1] = max(tmp1, key=tmp1.get)
    result = pop[0]
    idx = 0
    for i in range(len(result)):
        for j in range(len(result[0])):
            for k in range(len(result[0][0])):
                result[i][j][k] = list(ans[idx])
                idx += 1
    return result

def trans_pop(pop):
    pool = []
    for gene in pop:
        tmp = []
        for cell in gene:
            for node in cell:
                for element in node:
                    tmp.extend(element)
        pool.append(tmp)

    def transpose(matrix):
        new_matrix = []
        for i in range(len(matrix[0])):
            matrix1 = []
            for j in range(len(matrix)):
                matrix1.append(matrix[j][i])
            new_matrix.append(matrix1)
        return new_matrix
    pool = transpose(pool)
    return pool

def ratio(pop):
    pool = trans_pop(pop)
    diversity = []
    domination = []
    gene = []
    len_gene = len(pool[0])

    for gene_pos in pool:
        diver = len(set(gene_pos)) / len_gene
        diversity.append(diver)
        tmp = Counter(gene_pos).most_common(1)
        gene.append(tmp[0][0])
        domination.append(tmp[0][1] / len_gene)

    is_terminate = all([v > 0.6 for v in domination])

    return is_terminate, diversity, domination, gene

def cal_ent(x):
    """
    :param x: a list of gene at one pos
    :return: ent of x
    """
    x_value_list = list(set(x))
    ent = 0.
    x_dict = Counter(x)
    x_len = len(x)
    for v in x_value_list:
        p = x_dict[v] / x_len
        logp = np.log2(p)
        ent -= p * logp
    return ent

def cal_ent_pool(gene_pool):
    total_ent = 0
    for gene_pos in gene_pool:
        ent = cal_ent(gene_pos)
        total_ent += ent
    return total_ent
