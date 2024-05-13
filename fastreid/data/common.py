# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)


class KittiDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        left_pid_set = set()
        right_pid_set = set()
        left_cam_set = set()
        right_cam_set = set()
        for i in img_items:
            left_pid_set.add(i[0][1])
            left_cam_set.add(i[0][2])
            right_pid_set.add(i[1][1])
            right_cam_set.add(i[1][2])

        self.left_pids = sorted(list(left_pid_set))
        self.left_cams = sorted(list(left_cam_set))
        self.right_pids = sorted(list(right_pid_set))
        self.right_cams = sorted(list(right_cam_set))
        if relabel:
            self.left_pid_dict = dict([(p, i) for i, p in enumerate(self.left_pids)])
            self.right_pid_dict = dict([(p, i) for i, p in enumerate(self.right_pids)])
            self.left_cam_dict = dict([(p, i) for i, p in enumerate(self.left_cams)])
            self.right_cam_dict = dict([(p, i) for i, p in enumerate(self.right_cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        left_img_item = self.img_items[index][0]
        right_img_item = self.img_items[index][1]
        disp_item = self.img_items[index][2]
        object_class = self.img_items[index][3]
        x_min, y_min, focal, cx, cy = self.img_items[index][4]
        left_img_path = left_img_item[0]
        right_img_path = right_img_item[0]
        left_pid = left_img_item[1]
        right_pid = right_img_item[1]
        left_camid = left_img_item[2]
        right_camid = right_img_item[2]
        disp_path = disp_item[0]
        disp_offset = disp_item[1]
        left_img = read_image(left_img_path)
        right_img = read_image(right_img_path)
        # matlab处理的视差图扩大了100倍，现在复原
        disp_img = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH) / 100
        ori_width = disp_img.shape[1]
        ori_hw = disp_img.shape
        if object_class == 'Car':
            input_size = [224, 160]
        elif object_class == 'Pedestrian':
            input_size = [128, 256]
        elif object_class == 'Cyclist':
            input_size = [192, 192]
        else:
            raise ValueError("本项目仅提供Car、Pedestrian、Cyclist三个类别")
        # 得到原始深度图
        mask = (disp_img != 0)
        depth_img = np.zeros_like(disp_img, dtype=np.float32)
        depth_img[mask] = 1 / disp_img[mask]
        depth_image = scale_nonzero_values(depth_img)
        xyz_sampled = depth_to_point_cloud(depth_image, x_min, y_min, focal, cx, cy)

        # 由于图像resize，匹配网络求得视差也变化，需要真值和offset同比例变化，才能保持等式恒定
        disp_img = cv2.resize(disp_img, tuple(input_size), interpolation=cv2.INTER_NEAREST) * (input_size[0] / ori_width)
        disp_low_img = cv2.resize(disp_img, (disp_img.shape[1] // 4, disp_img.shape[0] // 4), interpolation=cv2.INTER_NEAREST)
        disp_offset = disp_offset * (input_size[0] / ori_width)
        disp_img = torch.tensor(disp_img, dtype=torch.float)
        disp_low_img = torch.tensor(disp_low_img, dtype=torch.float)
        if self.transform is not None:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        if self.relabel:
            left_pid = self.left_pid_dict[left_pid]
            right_pid = self.right_pid_dict[right_pid]
            assert left_pid == right_pid, "左目pid不等于右目"
            left_camid = self.left_cam_dict[left_camid]
            right_camid = self.right_cam_dict[right_camid]
        return {
            "left_images": left_img,
            "right_images": right_img,
            "ori_hw": ori_hw,
            "disp_images": disp_img,
            "disp_low_images": disp_low_img,
            "disp_offset": disp_offset,
            "targets": left_pid,
            "left_camids": left_camid,
            "right_camids": right_camid,
            "left_img_paths": left_img_path,
            "right_img_paths": right_img_path,
            "disp_img_paths": disp_path,
            "xyz_sampled": xyz_sampled
        }

    @property
    def num_classes(self):
        assert len(self.left_pids) == len(self.right_pids), "左目pid数量不等于右目"
        return len(self.left_pids)

    @property
    def num_cameras(self):
        return 2


def scale_nonzero_values(arr):
    nonzero_vals = arr[arr != 0]
    min_val = np.min(nonzero_vals)
    max_val = np.max(nonzero_vals)
    scaled_arr = arr.copy()
    scaled_arr[arr != 0] = 2 * ((arr[arr != 0] - min_val) / (max_val - min_val)) - 1
    return scaled_arr


def center_non_zero_coordinates(numpy_point_cloud):
    # Get the non-zero XYZ coordinates
    non_zero_mask = np.any(numpy_point_cloud != 0, axis=2)
    non_zero_xyz = numpy_point_cloud[non_zero_mask]

    # Compute the mean of the non-zero coordinates
    mean_xyz = np.mean(non_zero_xyz, axis=0)

    # Center the non-zero coordinates
    numpy_point_cloud[non_zero_mask] -= mean_xyz

    return numpy_point_cloud


def process_point_cloud(numpy_point_cloud, num_points=1024):
    # Create a mask of non-zero points 在第三个维度（xyz）判断是否为零
    non_zero_mask = np.any(numpy_point_cloud != 0, axis=2)

    # Get the non-zero points
    non_zero_xyz = numpy_point_cloud[non_zero_mask]

    # If there are not enough non-zero points, sample with replacement
    replace = non_zero_xyz.shape[0] < num_points

    # Sample num_points points randomly from the non-zero points
    idx = np.random.choice(non_zero_xyz.shape[0], num_points, replace=replace)

    xyz_sampled = non_zero_xyz[idx]
    return torch.tensor(xyz_sampled, dtype=torch.float)


def depth_to_point_cloud(depth_image, x_min, y_min, focal, cx, cy):
    # 深度图的高度和宽度
    height, width = depth_image.shape

    # 创建u, v坐标
    u = np.linspace(x_min, x_min + width - 1, width)
    v = np.linspace(y_min, y_min + height - 1, height)
    u, v = np.meshgrid(u, v)

    # 根据深度图和相机内参，计算X, Y, Z
    Z = depth_image
    X = (u - cx) * Z / focal
    Y = (v - cy) * Z / focal

    point_XYZ = np.concatenate((np.expand_dims(X, axis=2), np.expand_dims(Y, axis=2), np.expand_dims(Z, axis=2)), axis=2)
    point_XYZ = center_non_zero_coordinates(point_XYZ)
    xyz_sampled = process_point_cloud(point_XYZ)
    return xyz_sampled.transpose(0, 1)
