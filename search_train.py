import os
import sys
sys.path.append("..")
import time
import numpy as np
import glob
import allutils as utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from google_space_model import Model
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from genetype import POP


parser = argparse.ArgumentParser("EENAS on cifar10")
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=int, default=100, help='report frequency')
parser.add_argument('--gpu', default="0", type=str, help='gpu to run for single gpu')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')  # 600
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='EENAS on cifar10', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--cosepochs', type=int, default=600, help='CosineAnnealingLR epochs for one circle')
parser.add_argument('--nw', type=int, default=4, help='dataloader num workers')
parser.add_argument('--valid', type=int, default=1, help='dataloader num workers')
parser.add_argument('--T', type=int, default=25, help='dataloader num workers')
parser.add_argument('--layer_num', type=int, default=8, help='dataloader num workers')
parser.add_argument('--iters', type=int, default=800, help='dataloader num workers')

args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

trainloader = utils.get_dataloader_cifar10(args.data, args.batch_size, args.nw, 'train')
validloader = utils.get_dataloader_cifar10(args.data, args.batch_size, args.nw, 'valid')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
logging.info("args = %s", args)

if not torch.cuda.is_available():
    args.device = 'cpu'
    parallel_enable = False
else:
    gpu_str = ",".join(args.gpu)
    logging.info('gpu device = %s' % args.gpu)
    gpu_nums = len(list(args.gpu))
    parallel_enable = gpu_nums > 1
    gpu_ids = [int(x) for x in list(args.gpu)]
    args.batch_size *= gpu_nums
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    args.device = torch.device("cuda:{}".format(gpu_ids[0]))
    torch.cuda.set_device(args.device)


model = Model(C=16, class_num=10, cell_num=args.layer_num, node_num=4, stem_multiplier=3)
model.cuda()
logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
criterion = utils.SmoothingCrossEntropyLoss()
criterion = criterion.cuda()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iters)
# EA process
pop = POP(population=20, node_ids=[2,3,4,5], in_node_num=2, op_num=7, pm=0.1)
history = utils.ea_process_first(model, trainloader, validloader, criterion, optimizer, scheduler, args.iters, args.T, pop, args.save)




