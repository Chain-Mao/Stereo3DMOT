# from __future__ import print_function, division
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from torch.utils.data import DataLoader
import gc
sys.path.append('/home/simit/code/Stereo3DMOT')
from Fast_ACVNet import __models__, model_loss_train, model_loss_test
from Fast_ACVNet.utils import *
import cv2

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(
    description='Accurate and Efficient Stereo Matching via Attention Concatenation Volume (Fast-ACV)')
parser.add_argument('--model', default='Fast_ACVNet_plus', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--kitti15_datapath', default='/home/simit/code/Stereo3DMOT/data/KITTI/KITTI_2015', help='data path')
parser.add_argument('--trainlist', default='/home/simit/code/Stereo3DMOT/Fast_ACVNet/filenames/kitti15_train.txt', help='training list')
parser.add_argument('--testlist', default='/home/simit/code/Stereo3DMOT/Fast_ACVNet/filenames/kitti15_val.txt', help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=4, help='testing batch size')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
parser.add_argument('--lrepochs', default="300:10", type=str, help='the epochs to decay lr: the downscale rate')
parser.add_argument('--attention_weights_only', default=False, type=str, help='only train attention weights')
parser.add_argument('--logdir', default='./checkpoints', help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', default='', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.kitti15_datapath, args.trainlist, True)
test_dataset = StereoDataset(args.kitti15_datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=24, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=24, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp, args.attention_weights_only)
model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
# optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=0.9)
# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)
print("start at epoch {}".format(start_epoch))


def train():
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))

        # saving checkpoints

        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            # id_epoch = (epoch_idx + 1) % 100
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # # testing
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            # do_summary = global_step % args.summary_freq == 0
            do_summary = global_step % 1 == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                     batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        gc.collect()


# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()
    imgL, imgR, disp_gt, disp_gt_low = sample['left'], sample['right'], sample['disparity'], sample['disparity_low']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    disp_gt_low = disp_gt_low.cuda()
    optimizer.zero_grad()
    disp_ests = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    mask_low = (disp_gt_low < args.maxdisp) & (disp_gt_low > 0)
    masks = [mask, mask_low, mask, mask_low]
    disp_gts = [disp_gt, disp_gt_low, disp_gt, disp_gt_low]
    loss = model_loss_train(disp_ests, disp_gts, masks)
    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap"] = [disp_error_image_func.apply(disp_ests[0], disp_gt)]
            scalar_outputs["EPE"] = [EPE_metric(disp_ests[0], disp_gt, mask)]
            scalar_outputs["D1"] = [D1_metric(disp_ests[0], disp_gt, mask)]
            scalar_outputs["Thres1"] = [Thres_metric(disp_ests[0], disp_gt, mask, 1.0)]
            scalar_outputs["Thres2"] = [Thres_metric(disp_ests[0], disp_gt, mask, 2.0)]
            scalar_outputs["Thres3"] = [Thres_metric(disp_ests[0], disp_gt, mask, 3.0)]
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    disp_ests = model(imgL, imgR)
    disp_gts = [disp_gt]
    masks = [mask]
    loss = model_loss_test(disp_ests, disp_gts, masks)
    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    if compute_metrics:
        image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    train()
