# encoding: utf-8
import glob
import os.path as osp
import re
import warnings

import yaml

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Kitti(ImageDataset):
    dataset_name = "kitti"
    dataset_dir = ''

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, '/home/simit/code/Stereo3DMOT/data/KITTI/tracking/training/crop_image')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated.')

        with open('/home/simit/code/Stereo3DMOT/configs/Base-bagtricks.yml', 'r', encoding='utf-8') as f:
            file_content = f.read()
        content = yaml.load(file_content, yaml.FullLoader)
        ''' 选定目标类别 '''
        self.object_class = content.get('INPUT').get('CLASS')

        self.train_left_dir = osp.join(self.data_dir, 'image_02/' + self.object_class)
        self.test_left_dir = osp.join(self.data_dir, 'image_02/' + self.object_class)
        self.train_right_dir = osp.join(self.data_dir, 'image_03/' + self.object_class)
        self.test_right_dir = osp.join(self.data_dir, 'image_03/' + self.object_class)

        self.disp_dir = osp.join(self.data_dir, 'disp/' + self.object_class)

        self.train_dir = [self.train_left_dir, self.train_right_dir, self.disp_dir]
        self.test_dir = [self.test_left_dir, self.test_right_dir, self.disp_dir]

        required_files = [
            self.train_left_dir,
            self.test_left_dir,
            self.train_right_dir,
            self.test_right_dir
        ]
        self.check_before_run(required_files)

        train = lambda: self.process_dir(self.train_dir)
        query = lambda: self.process_dir(self.test_dir, is_train=False)
        gallery = lambda: self.process_dir(self.test_dir, is_train=False)

        super(Kitti, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        left_img_paths = glob.glob(osp.join(dir_path[0], '*.png'))
        right_img_paths = glob.glob(osp.join(dir_path[1], '*.png'))
        disp_paths = glob.glob(osp.join(dir_path[2], '*.tif'))

        data = []
        left_img_paths.sort()
        right_img_paths.sort()
        disp_paths.sort()
        for left_img_path, right_img_path, disp_path in zip(left_img_paths, right_img_paths, disp_paths):
            right_pid, right_camid, _, _, _, _, _, _, _, _ = re.findall(r'(\d+)_c(\d+)s(\d+)_f(\d+)_dis(\d+)_x(\d+)_y(\d+)_focal([\d.]+)_cx([\d.]+)_cy([\d.]+)\.png', right_img_path)[0]
            left_pid, left_camid, seq, frame, disp_offset, x_min, y_min, focal, cx, cy = re.findall(r'(\d+)_c(\d+)s(\d+)_f(\d+)_dis(\d+)_x(\d+)_y(\d+)_focal([\d.]+)_cx([\d.]+)_cy([\d.]+)\.png', left_img_path)[0]
            left_pid, left_camid, right_pid, right_camid, disp_offset, x_min, y_min, focal, cx, cy = int(left_pid), int(left_camid), int(right_pid), int(right_camid), float(disp_offset), int(x_min), int(y_min), float(focal), float(cx), float(cy)
            if is_train:
                left_pid = self.dataset_name + "_" + str(left_pid)
                left_camid = self.dataset_name + "_" + str(left_camid)
                right_pid = self.dataset_name + "_" + str(right_pid)
                right_camid = self.dataset_name + "_" + str(right_camid)
            data.append([(left_img_path, left_pid, left_camid), (right_img_path, right_pid, right_camid), (disp_path, disp_offset), self.object_class, (x_min, y_min, focal, cx, cy)])

        return data

# @DATASET_REGISTRY.register()
# class Kitti(ImageDataset):
#     dataset_name = "kitti"
#     dataset_dir = ''
#
#     def __init__(self, root='datasets', **kwargs):
#         # self.root = osp.abspath(osp.expanduser(root))
#         self.root = root
#         self.dataset_dir = osp.join(self.root, self.dataset_dir)
#
#         # allow alternative directory structure
#         self.data_dir = self.dataset_dir
#         data_dir = osp.join(self.data_dir, '/home/simit/code/Stereo3DMOT/data/KITTI/tracking/training/crop_image')
#         if osp.isdir(data_dir):
#             self.data_dir = data_dir
#         else:
#             warnings.warn('The current data structure is deprecated.')
#
#         with open('/home/simit/code/Stereo3DMOT/configs/Base-bagtricks.yml', 'r', encoding='utf-8') as f:
#             file_content = f.read()
#         content = yaml.load(file_content, yaml.FullLoader)
#         ''' 选定目标类别 '''
#         object_class = content.get('INPUT').get('CLASS')
#
#         self.train_left_dir = osp.join(self.data_dir, 'image_02/' + object_class)
#         self.train_right_dir = osp.join(self.data_dir, 'image_03/' + object_class)
#         self.train_dir = [self.train_left_dir, self.train_right_dir]
#
#         required_files = [
#             self.train_left_dir,
#             self.train_right_dir,
#         ]
#         self.check_before_run(required_files)
#
#         train = self.process_dir(self.train_left_dir, self.train_right_dir)
#
#         super().__init__(train, [], [], **kwargs)
#
#     def process_dir(self, train_left_dir, train_right_dir, is_train=True):
#         left_img_paths = glob.glob(osp.join(train_left_dir, '*.png'))
#         right_img_paths = glob.glob(osp.join(train_right_dir, '*.png'))
#         pattern = re.compile(r'([-\d]+)_c(\d)')
#
#         data = []
#         left_img_paths.sort()
#         right_img_paths.sort()
#         for left_img_path, right_img_path in zip(left_img_paths, right_img_paths):
#             left_pid, left_camid = map(int, pattern.search(left_img_path).groups())
#             right_pid, right_camid = map(int, pattern.search(right_img_path).groups())
#             if is_train:
#                 left_pid = self.dataset_name + "_" + str(left_pid)
#                 left_camid = self.dataset_name + "_" + str(left_camid)
#                 right_pid = self.dataset_name + "_" + str(right_pid)
#                 right_camid = self.dataset_name + "_" + str(right_camid)
#             data.append([(left_img_path, left_pid, left_camid), (right_img_path, right_pid, right_camid)])
#
#         return data