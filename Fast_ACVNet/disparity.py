from __future__ import print_function, division
import argparse
import os
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
from tqdm import tqdm, trange
from Fast_ACVNet import __models__
from torch.utils.data import DataLoader
import gc
import skimage
import skimage.io
import cv2
from PIL import Image
from torchvision import transforms
# cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Accurate and Efficient Stereo Matching via Attention Concatenation Volume (Fast-ACV)')
parser.add_argument('--model', default='Fast_ACVNet_plus', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--datapath', default="/data/Middlebury/", help='data path')
parser.add_argument('--resolution', type=str, default='H')
parser.add_argument('--loadckpt', default='/home/simit/code/Stereo3DMOT/Fast_ACVNet/weights/kitti_2015.ckpt', help='load the weights from a specific checkpoint')

# parse arguments
args = parser.parse_args()
model = __models__[args.model](args.maxdisp, False)
model = nn.DataParallel(model)
model.cuda()
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])
model.eval()


def disparity_computer(left_img, right_img):

    limg = Image.open(limg_path).convert('RGB')
    rimg = Image.open(rimg_path).convert('RGB')

    w, h = limg.size
    wi, hi = (w // 32 + 1) * 32, (h // 32 + 1) * 32

    limg = limg.crop((w - wi, h - hi, w, h))
    rimg = rimg.crop((w - wi, h - hi, w, h))

    limg_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(limg)
    rimg_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(rimg)
    limg_tensor = limg_tensor.unsqueeze(0).cuda()
    rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

    with torch.no_grad():
        pred_disp  = model(limg_tensor, rimg_tensor)[-1]

        pred_disp = pred_disp[:, hi - h:, wi - w:]

    pred_np = pred_disp.squeeze().cpu().numpy()

    torch.cuda.empty_cache()

    return pred_np