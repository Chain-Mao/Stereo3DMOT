import numpy as np
from numba import jit
from copy import deepcopy
from .kitti_oxts import roty
import csv
import math as m
import cv2


class Box3D:
    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, ry=None, s=None):
        self.x = x      # center x
        self.y = y      # center y
        self.z = z      # center z
        self.h = h      # height
        self.w = w      # width
        self.l = l      # length
        self.ry = ry    # orientation
        self.s = s      # detection score
        self.corners_3d_cam = None
        self.feature = None
        self.depth = None
        self.disp_offset = None
        self.hw = None

    def __str__(self):
        return 'x: {}, y: {}, z: {}, heading: {}, length: {}, width: {}, height: {}, score: {}'.format(
            self.x, self.y, self.z, self.ry, self.l, self.w, self.h, self.s)
    
    @classmethod
    def bbox2dict(cls, bbox):
        return {
            'center_x': bbox.x, 'center_y': bbox.y, 'center_z': bbox.z,
            'height': bbox.h, 'width': bbox.w, 'length': bbox.l, 'heading': bbox.ry}
    
    @classmethod
    def bbox2array(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h])
        else:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h, bbox.s])

    @classmethod
    def bbox2array_raw(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry])
        else:
            return np.array([bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry, bbox.s])

    @classmethod
    def array2bbox_raw(cls, data):
        # take the format of data of [h,w,l,x,y,z,theta]

        bbox = Box3D()
        bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox
    
    @classmethod
    def array2bbox(cls, data):
        # take the format of data of [x,y,z,theta,l,w,h]

        bbox = Box3D()
        bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox
    
    @classmethod
    def box2corners3d_camcoord(cls, bbox):
        ''' Takes an object's 3D box with the representation of [x,y,z,theta,l,w,h] and
            convert it to the 8 corners of the 3D box, the box is in the camera coordinate
            with right x, down y, front z

            Returns:
                corners_3d: (8,3) array in rect camera coord

            box corner order is like follows
                    1 -------- 0         top is bottom because y direction is negative
                   /|         /|
                  2 -------- 3 .
                  | |        | |
                  . 5 -------- 4
                  |/         |/
                  6 -------- 7

            rect/ref camera coord:
            right x, down y, front z

            x -> w, z -> l, y -> h
        '''

        # if already computed before, then skip it
        if bbox.corners_3d_cam is not None:
            return bbox.corners_3d_cam

        # compute rotational matrix around yaw axis
        # -1.57 means straight, so there is a rotation here
        R = roty(bbox.ry)

        # 3d bounding box dimensions
        l, w, h = bbox.l, bbox.w, bbox.h

        # 3d bounding box corners
        x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
        y_corners = [0,0,0,0,-h,-h,-h,-h]
        z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]

        # rotate and translate 3d bounding box
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0,:] = corners_3d[0,:] + bbox.x
        corners_3d[1,:] = corners_3d[1,:] + bbox.y
        corners_3d[2,:] = corners_3d[2,:] + bbox.z
        corners_3d = np.transpose(corners_3d)
        bbox.corners_3d_cam = corners_3d

        return corners_3d


def NormalizeVector(P):
    return np.append(P, [1])


def E2R(Ry, Rx, Rz):
    '''Combine Euler angles to the rotation matrix (right-hand)

        Inputs:
            Ry, Rx, Rz : rotation angles along  y, x, z axis
                         only has Ry in the KITTI dataset
        Returns:
            3 x 3 rotation matrix

    '''
    R_yaw = np.array([[m.cos(Ry), 0, m.sin(Ry)],
                      [0, 1, 0],
                      [-m.sin(Ry), 0, m.cos(Ry)]])
    R_pitch = np.array([[1, 0, 0],
                        [0, m.cos(Rx), -m.sin(Rx)],
                        [0, m.sin(Rx), m.cos(Rx)]])
    # R_roll = np.array([[[m.cos(Rz), -m.sin(Rz), 0],
    #                    [m.sin(Rz), m.cos(Rz), 0],
    #                    [ 0,         0 ,     1]])
    return (R_pitch.dot(R_yaw))


class KittiObject:
    def __init__(self):
        self.cls = ''  # Car, Van, Truck
        self.alpha = 0  # viewpoint angle -pi - pi
        self.boxes = (Box2d(), \
                      Box2d(), Box2d())  # Box2d list, default order: box_left, box_right, box_merge
        self.pos = []  # x, y, z in cam2 frame
        self.dim = []  # width(x), height(y), length(z)
        self.orientation = 0  # [-pi - pi]
        self.R = []  # rotation matrix in cam2 frame


class Box2d:
    def __init__(self):
        self.box = []  # left, top, right bottom in 2D image
        self.keypoints = []  # holds the u coordinates of 4 keypoints, -1 denotes the invisible one
        self.visible_left = 0  # The left side is visible (not occluded) by other object
        self.visible_right = 0  # The right side is visible (not occluded) by other object


class FrameCalibrationData:
    '''Frame Calibration Holder
        p0-p3      Camera P matrix. Contains extrinsic 3x4
                   and intrinsic parameters.
        r0_rect    Rectification matrix, required to transform points 3x3
                   from velodyne to camera coordinate frame.
        tr_velodyne_to_cam0     Used to transform from velodyne to cam 3x4
                                coordinate frame according to:
                                Point_Camera = P_cam * R0_rect *
                                                Tr_velo_to_cam *
                                                Point_Velodyne.
    '''

    def __init__(self):
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p2_3 = []
        self.r0_rect = []
        self.t_cam2_cam0 = []
        self.tr_velodyne_to_cam0 = []


def Space2Image(P0, pts3):
    pts2_norm = P0.dot(pts3)
    pts2 = np.array([int(pts2_norm[0] / pts2_norm[2]), int(pts2_norm[1] / pts2_norm[2])])
    return pts2


def read_obj_calibration(CALIB_PATH):
    ''' Reads in Calibration file from Kitti Dataset.

        Inputs:
        CALIB_PATH : Str PATH of the calibration file.

        Returns:
        frame_calibration_info : FrameCalibrationData
                                Contains a frame's full calibration data.
        ^ z        ^ z                                      ^ z         ^ z
        | cam2     | cam0                                   | cam3      | cam1
        |-----> x  |-----> x                                |-----> x   |-----> x

    '''
    frame_calibration_info = FrameCalibrationData()

    data_file = open(CALIB_PATH, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all = []

    for i in range(4):
        p = data[i]
        p = p[1:][0:12]
        p = [float(p[i]) for i in range(len(p))]
        p = np.reshape(p, (3, 4))
        p_all.append(p)

    # based on camera 0
    frame_calibration_info.p0 = p_all[0]
    frame_calibration_info.p1 = p_all[1]
    frame_calibration_info.p2 = p_all[2]
    frame_calibration_info.p3 = p_all[3]

    # based on camera 2
    frame_calibration_info.p2_2 = np.copy(p_all[2])
    frame_calibration_info.p2_2[0, 3] = frame_calibration_info.p2_2[0, 3] - frame_calibration_info.p2[0, 3]

    frame_calibration_info.p2_3 = np.copy(p_all[3])
    frame_calibration_info.p2_3[0, 3] = frame_calibration_info.p2_3[0, 3] - frame_calibration_info.p2[0, 3]

    frame_calibration_info.t_cam2_cam0 = np.zeros(3)
    frame_calibration_info.t_cam2_cam0[0] = (frame_calibration_info.p2[0, 3] - frame_calibration_info.p0[0, 3]) / \
                                            frame_calibration_info.p2[0, 0]

    # Read in rectification matrix
    tr_rect = data[4]
    tr_rect = tr_rect[1:][0:9]
    tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
    frame_calibration_info.r0_rect = np.reshape(tr_rect, (3, 3))

    # Read in velodyne to cam matrix
    tr_v2c = data[5]
    tr_v2c = tr_v2c[1:][0:12]
    tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
    frame_calibration_info.tr_velodyne_to_cam0 = np.reshape(tr_v2c, (3, 4))

    return frame_calibration_info


def read_obj_data(obj, calib=None, im_shape=None):
    """Reads in object label file from Kitti Object Dataset.

        Inputs:
            LABEL_PATH : Str PATH of the label file.
        Returns:
            List of KittiObject : Contains all the labeled data

    """
    # x, y, z, rot_y, l, w, h
    obj = Box3D.bbox2array(obj)
    object_it = KittiObject()
    #                            width          height         lenth
    object_it.dim = np.array([obj[5], obj[6], obj[4]]).astype(float)

    # The KITTI GT 3D box is on cam0 frame, while we deal with cam2 frame
    object_it.pos = np.array(obj[0:3]).astype(float)  # 0.062 这里应该不需要+t_cam2_cam0 存疑
    # The orientation definition is inconsitent with right-hand coordinates in kitti
    object_it.orientation = float(obj[3]) + m.pi / 2
    object_it.R = E2R(object_it.orientation, 0, 0)

    pts3_c_o = []  # 3D location of 3D bounding box corners ## 3d bbox的8个顶点
    pts3_c_o.append(object_it.pos + object_it.R.dot([-object_it.dim[0], 0, -object_it.dim[2]]) / 2.0)
    pts3_c_o.append(object_it.pos + object_it.R.dot([-object_it.dim[0], 0, object_it.dim[2]]) / 2.0)
    pts3_c_o.append(object_it.pos + object_it.R.dot([object_it.dim[0], 0, object_it.dim[2]]) / 2.0)
    pts3_c_o.append(object_it.pos + object_it.R.dot([object_it.dim[0], 0, -object_it.dim[2]]) / 2.0)

    pts3_c_o.append(
        object_it.pos + object_it.R.dot([-object_it.dim[0], -2.0 * object_it.dim[1], -object_it.dim[2]]) / 2.0)
    pts3_c_o.append(
        object_it.pos + object_it.R.dot([-object_it.dim[0], -2.0 * object_it.dim[1], object_it.dim[2]]) / 2.0)
    pts3_c_o.append(
        object_it.pos + object_it.R.dot([object_it.dim[0], -2.0 * object_it.dim[1], object_it.dim[2]]) / 2.0)
    pts3_c_o.append(
        object_it.pos + object_it.R.dot([object_it.dim[0], -2.0 * object_it.dim[1], -object_it.dim[2]]) / 2.0)

    object_it.boxes[0].box = np.array([10000, 10000, 0, 0]).astype(float)
    object_it.boxes[1].box = np.array([10000, 10000, 0, 0]).astype(float)
    object_it.boxes[2].box = np.array([0.0, 0.0, 0.0, 0.0]).astype(float)
    object_it.boxes[0].keypoints = np.array([-1.0, -1.0, -1.0, -1.0]).astype(float)
    object_it.boxes[1].keypoints = np.array([-1.0, -1.0, -1.0, -1.0]).astype(float)
    for j in range(2):  # left and right boxes
        for i in range(8):
            if pts3_c_o[i][2] < 0:
                continue
            if j == 0:  # project 3D corner to left image
                pt2 = Space2Image(calib.p2_2, NormalizeVector(pts3_c_o[i]))
            elif j == 1:  # project 3D corner to right image
                pt2 = Space2Image(calib.p2_3, NormalizeVector(pts3_c_o[i]))
            if i < 4:
                object_it.boxes[j].keypoints[i] = pt2[0]

            object_it.boxes[j].box[0] = min(object_it.boxes[j].box[0], pt2[0])
            object_it.boxes[j].box[1] = min(object_it.boxes[j].box[1], pt2[1])
            object_it.boxes[j].box[2] = max(object_it.boxes[j].box[2], pt2[0])
            object_it.boxes[j].box[3] = max(object_it.boxes[j].box[3], pt2[1])

        object_it.boxes[j].box[0] = max(object_it.boxes[j].box[0], 0)
        object_it.boxes[j].box[1] = max(object_it.boxes[j].box[1], 0)

        object_it.boxes[j].box[2] = min(object_it.boxes[j].box[2], im_shape[1] - 1)
        object_it.boxes[j].box[3] = min(object_it.boxes[j].box[3], im_shape[0] - 1)

        # deal with unvisible keypoints
        left_keypoint, right_keypoint = 5000, 0
        left_inx, right_inx = -1, -1
        # 1. Select keypoints that lie on the left and right side of the 2D box
        for i in range(4):
            if object_it.boxes[j].keypoints[i] < left_keypoint:
                left_keypoint = object_it.boxes[j].keypoints[i]
                left_inx = i
            if object_it.boxes[j].keypoints[i] > right_keypoint:
                right_keypoint = object_it.boxes[j].keypoints[i]
                right_inx = i
        # 2. For keypoints between left and right side, select the visible one
        for i in range(4):
            if i == left_inx or i == right_inx:
                continue
            if pts3_c_o[i][2] > object_it.pos[2]:
                object_it.boxes[j].keypoints[i] = -1

    # calculate the union of the left and right box
    object_it.boxes[2].box[0] = min(object_it.boxes[1].box[0], object_it.boxes[0].box[0])
    object_it.boxes[2].box[1] = min(object_it.boxes[1].box[1], object_it.boxes[0].box[1])
    object_it.boxes[2].box[2] = max(object_it.boxes[1].box[2], object_it.boxes[0].box[2])
    object_it.boxes[2].box[3] = max(object_it.boxes[1].box[3], object_it.boxes[0].box[3])

    return object_it


# def project_to_image(x3d, P):
#     # 将3D点投影到2D图像中
#     x3d_h = np.concatenate([x3d, np.ones((1, x3d.shape[1]))], axis=0)
#     x2d_h = np.dot(P, x3d_h)
#     x2d_h /= x2d_h[2]
#     return x2d_h[:2]
#
#
# def compute_2d_left_right(bbox3d, P2, P3, R_rect):
#     # 将3D检测框转换到0号相机坐标系,不经过点云坐标系
#     R0_rect = np.eye(4, dtype=R_rect.dtype)
#     R0_rect[:3, :3] = R_rect
#     R = np.array([[np.cos(bbox3d[3]), 0, np.sin(bbox3d[3])],
#                   [0, 1, 0],
#                   [-np.sin(bbox3d[3]), 0, np.cos(bbox3d[3])]])
#     l, w, h = bbox3d[4], bbox3d[5], bbox3d[6]
#     x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
#     y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
#     z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
#     corners_3d = np.vstack([x_corners, y_corners, z_corners])
#     corners_3d = np.dot(R, corners_3d)
#     corners_3d[0, :] += bbox3d[0]
#     corners_3d[1, :] += bbox3d[1]
#     corners_3d[2, :] += bbox3d[2]
#     corners_3d = np.dot(R0_rect, np.vstack([corners_3d, np.ones(8)]))
#     corners_3d = corners_3d[:3, :]
#
#     # 将3D检测框投影到2号相机图像中
#     corners_2d_camera2 = project_to_image(corners_3d, P2)
#     # 将3D检测框投影到3号相机图像中
#     corners_2d_camera3 = project_to_image(corners_3d, P3)
#
#     # 计算2D检测框的坐标
#     x2min = max(int(np.min(corners_2d_camera2[0])), 0)
#     y2min = max(int(np.min(corners_2d_camera2[1])), 0)
#     x2max = max(int(np.max(corners_2d_camera2[0])), 0)
#     y2max = max(int(np.max(corners_2d_camera2[1])), 0)
#
#     x3min = max(int(np.min(corners_2d_camera3[0])), 0)
#     y3min = max(int(np.min(corners_2d_camera3[1])), 0)
#     x3max = max(int(np.max(corners_2d_camera3[0])), 0)
#     y3max = max(int(np.max(corners_2d_camera3[1])), 0)
#
#     ymin = min(y2min, y3min)
#     ymax = max(y2max, y3max)
#     return [x2min, ymin, x2max, ymax], [x3min, ymin, x3max, ymax]
