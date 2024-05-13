# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import cv2
import torch
from PIL import Image
import numpy as np, os, copy, math
from AB3DMOT_libs.box import Box3D
from AB3DMOT_libs.matching import data_association, obj_get_feature
from AB3DMOT_libs.kalman_filter import KF, TrackState
from AB3DMOT_libs.vis import vis_obj
from Xinshuo_PyToolbox.xinshuo_miscellaneous import print_log
from Xinshuo_PyToolbox.xinshuo_io import mkdir_if_missing
from AB3DMOT_libs.vis import draw_box3d_image
from Xinshuo_PyToolbox.xinshuo_visualization import random_colors

from fastreid.predictor import FeatureExtraction
from fastreid.config import get_cfg

np.set_printoptions(suppress=True, precision=3)


# A Baseline of 3D Multi-Object Tracking
class AB3DMOT(object):
    def __init__(self, cfg, cat, calib=None, calib_reid=None, oxts=None, img_dir=None, img_right_dir=None, vis_dir=None,
                 hw=None, log=None, ID_init=0):

        # vis and log purposes
        self.img_dir = img_dir
        self.img_right_dir = img_right_dir
        self.vis_dir = vis_dir
        self.vis = cfg.vis
        self.hw = hw
        self.log = log

        # counter
        self.trackers = []
        self.frame_count = 0
        self.ID_count = [ID_init]
        self.id_now_output = []

        # config
        self.cat = cat
        self.ego_com = cfg.ego_com  # ego motion compensation
        self.split = cfg.split
        self.calib = calib
        self.calib_reid = calib_reid
        self.oxts = oxts
        self.affi_process = cfg.affi_pro  # post-processing affinity
        self.get_param(cfg, cat)
        self.print_param()

        # reid config
        self.reid_config_file = cfg.reid_config_file
        self.reid_opts = cfg.reid_opts
        self.reid_cfg = self.setup_reid_cfg()
        # use multiprocess for feature extraction
        self.feature_extraction = FeatureExtraction(self.reid_cfg, parallel=False)

        # debug
        # self.debug_id = 2
        self.debug_id = None

    def setup_reid_cfg(self):
        # load config from file and command-line arguments
        cfg = get_cfg()
        # add_partialreid_config(cfg)
        cfg.merge_from_file(self.reid_config_file)
        cfg.merge_from_list(self.reid_opts)
        cfg.freeze()
        return cfg

    def get_param(self, cfg, cat):
        # get parameters for each dataset
        if cfg.dataset == 'KITTI':
            if cfg.det_name == 'pvrcnn':  # tuned for PV-RCNN detections
                if cat == 'Car':
                    algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 2
                elif cat == 'Pedestrian':
                    algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 4
                elif cat == 'Cyclist':
                    algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 2, 3, 4
                else:
                    assert False, 'error'

            # 创新点: EIOU的改进"eiou_3d"
            elif cfg.det_name == 'pointrcnn':  # tuned for PointRCNN detections
                if cat == 'Car':
                    algm, metric, thres, min_hits, unconfirm_max_age, confirm_max_age = 'hungar', 'eiou_3d', -0.2, 2, 3, 22
                    # algm, metric, thres, min_hits, unconfirm_max_age, confirm_max_age = 'hungar', 'giou_3d', -0.2, 3, 3, 22
                elif cat == 'Pedestrian':
                    algm, metric, thres, min_hits, unconfirm_max_age, confirm_max_age = 'greedy', 'eiou_3d', -0.1, 1, 4, 23
                elif cat == 'Cyclist':
                    algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', 2, 3, 4
                else:
                    assert False, 'error'

            elif cfg.det_name == 'deprecated':
                if cat == 'Car':
                    algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 6, 3, 2
                elif cat == 'Pedestrian':
                    algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 1, 3, 2
                elif cat == 'Cyclist':
                    algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 6, 3, 2
                else:
                    assert False, 'error'
            else:
                assert False, 'error'

        elif cfg.dataset == 'nuScenes':
            if cfg.det_name == 'centerpoint':  # tuned for CenterPoint detections
                if cat == 'Car':
                    algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 2
                elif cat == 'Pedestrian':
                    algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.5, 1, 2
                elif cat == 'Truck':
                    algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 2
                elif cat == 'Trailer':
                    algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.3, 3, 2
                elif cat == 'Bus':
                    algm, metric, thres, min_hits, max_age = 'greedy', 'eiou_3d', -0.4, 1, 2
                elif cat == 'Motorcycle':
                    algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.7, 3, 2
                elif cat == 'Bicycle':
                    algm, metric, thres, min_hits, max_age = 'greedy', 'gist_3d', 6, 3, 2
                else:
                    assert False, 'error'
            elif cfg.det_name == 'megvii':  # tuned for Megvii detections
                if cat == 'Car':
                    algm, metric, thres, min_hits, max_age = 'greedy', 'eiou_3d', -0.5, 1, 2
                elif cat == 'Pedestrian':
                    algm, metric, thres, min_hits, max_age = 'greedy', 'dist_3d', 2, 1, 2
                elif cat == 'Truck':
                    algm, metric, thres, min_hits, max_age = 'greedy', 'eiou_3d', -0.2, 1, 2
                elif cat == 'Trailer':
                    algm, metric, thres, min_hits, max_age = 'greedy', 'eiou_3d', -0.2, 3, 2
                elif cat == 'Bus':
                    algm, metric, thres, min_hits, max_age = 'greedy', 'eiou_3d', -0.2, 1, 2
                elif cat == 'Motorcycle':
                    algm, metric, thres, min_hits, max_age = 'greedy', 'eiou_3d', -0.8, 3, 2
                elif cat == 'Bicycle':
                    algm, metric, thres, min_hits, max_age = 'greedy', 'eiou_3d', -0.6, 3, 2
                else:
                    assert False, 'error'
            elif cfg.det_name == 'deprecated':
                if cat == 'Car':
                    metric, thres, min_hits, max_age = 'dist', 10, 3, 2
                elif cat == 'Pedestrian':
                    metric, thres, min_hits, max_age = 'dist', 6, 3, 2
                elif cat == 'Bicycle':
                    metric, thres, min_hits, max_age = 'dist', 6, 3, 2
                elif cat == 'Motorcycle':
                    metric, thres, min_hits, max_age = 'dist', 10, 3, 2
                elif cat == 'Bus':
                    metric, thres, min_hits, max_age = 'dist', 10, 3, 2
                elif cat == 'Trailer':
                    metric, thres, min_hits, max_age = 'dist', 10, 3, 2
                elif cat == 'Truck':
                    metric, thres, min_hits, max_age = 'dist', 10, 3, 2
                else:
                    assert False, 'error'
            else:
                assert False, 'error'
        else:
            assert False, 'no such dataset'

        # add negative due to it is the cost
        if metric in ['dist_3d', 'dist_2d', 'm_dis']: thres *= -1
        self.algm, self.metric, self.thres, self.unconfirm_max_age, self.confirm_max_age, self.min_hits = \
            algm, metric, thres, unconfirm_max_age, confirm_max_age, min_hits

        # define max/min values for the output affinity matrix
        if self.metric in ['dist_3d', 'dist_2d', 'm_dis']:
            self.max_sim, self.min_sim = 0.0, -100.
        elif self.metric in ['iou_2d', 'iou_3d']:
            self.max_sim, self.min_sim = 1.0, 0.0
        elif self.metric in ['giou_2d', 'giou_3d']:
            self.max_sim, self.min_sim = 1.0, -1.0
        elif self.metric in ['eiou_3d']:
            self.max_sim, self.min_sim = 1.0, -1.0

    def print_param(self):
        print_log('\n\n***************** Parameters for %s *********************' % self.cat, log=self.log,
                  display=False)
        print_log('matching algorithm is %s' % self.algm, log=self.log, display=False)
        print_log('distance metric is %s' % self.metric, log=self.log, display=False)
        print_log('distance threshold is %f' % self.thres, log=self.log, display=False)
        print_log('min hits is %f' % self.min_hits, log=self.log, display=False)
        print_log('unconfirm max age is %f' % self.unconfirm_max_age, log=self.log, display=False)
        print_log('confirm max age is %f' % self.confirm_max_age, log=self.log, display=False)
        print_log('ego motion compensation is %d' % self.ego_com, log=self.log, display=False)

    def process_dets(self, dets):
        # convert each detection into the class Box3D
        # inputs:
        # 	dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]

        dets_new = []
        for det in dets:
            det_tmp = Box3D.array2bbox_raw(det)
            dets_new.append(det_tmp)

        return dets_new

    def within_range(self, theta):
        # make sure the orientation is within a proper range

        if theta >= np.pi: theta -= np.pi * 2  # make the theta still in the range
        if theta < -np.pi: theta += np.pi * 2

        return theta

    def orientation_correction(self, theta_pre, theta_obs):
        # update orientation in propagated tracks and detected boxes so that they are within 90 degree

        # make the theta still in the range
        theta_pre = self.within_range(theta_pre)
        theta_obs = self.within_range(theta_obs)

        # if the angle of two theta is not acute angle, then make it acute
        if abs(theta_obs - theta_pre) > np.pi / 2.0 and abs(theta_obs - theta_pre) < np.pi * 3 / 2.0:
            theta_pre += np.pi
            theta_pre = self.within_range(theta_pre)

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
            if theta_obs > 0:
                theta_pre += np.pi * 2
            else:
                theta_pre -= np.pi * 2

        return theta_pre, theta_obs

    def ego_motion_compensation(self, frame, trks):
        # inverse ego motion compensation, move trks from the last frame of coordinate to the current frame for matching

        from AB3DMOT_libs.kitti_oxts import get_ego_traj, egomotion_compensation_ID
        assert len(self.trackers) == len(trks), 'error'
        ego_xyz_imu, ego_rot_imu, left, right = get_ego_traj(self.oxts, frame, 1, 1, only_fut=True, inverse=True)
        for index in range(len(self.trackers)):
            trk_tmp = trks[index]
            xyz = np.array([trk_tmp.x, trk_tmp.y, trk_tmp.z]).reshape((1, -1))
            compensated = egomotion_compensation_ID(xyz, self.calib, ego_rot_imu, ego_xyz_imu, left, right)
            trk_tmp.x, trk_tmp.y, trk_tmp.z = compensated[0]
            # tracks同样操作
            self.trackers[index].box.x, self.trackers[index].box.y, self.trackers[index].box.z = compensated[0]
            # update compensated state in the Kalman filter
            try:
                self.trackers[index].kf.x[:3] = copy.copy(compensated).reshape((-1))
            except:
                self.trackers[index].kf.x[:3] = copy.copy(compensated).reshape((-1, 1))

        return trks

    def visualization(self, img, dets, trks, calib, hw, save_path, height_threshold=0):
        # visualize to verify if the ego motion compensation is done correctly
        # ideally, the ego-motion compensated tracks should overlap closely with detections
        dets, trks = copy.copy(dets), copy.copy(trks)
        img = np.array(Image.open(img))
        max_color = 20
        colors = random_colors(max_color)  # Generate random colors

        # visualize all detections as yellow boxes
        for det_tmp in dets:
            img = vis_obj(det_tmp, img, calib, hw, (255, 255, 0))  # yellow for detection

        # visualize color-specific tracks
        count = 0
        ID_list = [tmp.id for tmp in self.trackers]
        for trk_tmp in trks:
            ID_tmp = ID_list[count]
            color_float = colors[int(ID_tmp) % max_color]
            color_int = tuple([int(tmp * 255) for tmp in color_float])
            str_vis = '%d, %f' % (ID_tmp, trk_tmp.o)
            img = vis_obj(trk_tmp, img, calib, hw, color_int, str_vis)  # blue for tracklets
            count += 1

        img = Image.fromarray(img)
        img = img.resize((hw['image'][1], hw['image'][0]))
        img.save(save_path)

    # 这段代码实现了一个目标跟踪器的预测功能。该函数将从现有跟踪器中获取预测位置，然后使用卡尔曼滤波器进行位置预测，同时更新跟踪器的统计信息。
    def prediction(self):
        # get predicted locations from existing tracks

        trks = []
        for t in range(len(self.trackers)):

            # propagate locations
            kf_tmp = self.trackers[t]
            if kf_tmp.id == self.debug_id:
                print('\n before prediction')
                print(kf_tmp.kf.x.reshape((-1)))
                print('\n current velocity')
                print(kf_tmp.get_velocity())
            kf_tmp.kf.predict()
            if kf_tmp.id == self.debug_id:
                print('After prediction')
                print(kf_tmp.kf.x.reshape((-1)))
            kf_tmp.kf.x[3] = self.within_range(kf_tmp.kf.x[3])

            # update statistics
            kf_tmp.time_since_update += 1
            trk_tmp = kf_tmp.kf.x.reshape((-1))[:7]
            trks.append(Box3D.array2bbox(trk_tmp))

            # 轨迹预测框
            self.trackers[t].box = Box3D.array2bbox(trk_tmp)

        return trks

    def update(self, matched, unmatched_trks, dets, info):
        # update matched trackers with assigned detections
        dets = copy.copy(dets)
        # self.trackers是当前所有的跟踪器列表，t是跟踪器的索引，trk是对应的跟踪器对象。
        # 对于没有匹配上的跟踪器，unmatched_trks中记录了对应的索引，需要将这些跟踪器的time_since_update属性加1，表示它们已经跟丢了目标的时间增加了。
        # 而对于匹配上的跟踪器，需要更新其状态，即将其预测的位置更新为检测到的位置，同时将time_since_update属性重置为0，表示刚刚更新过状态。
        # 同时，hits属性加1，表示目标被成功跟踪的次数增加了。
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]  # a list of index
                assert len(d) == 1, 'error'

                # update statistics
                trk.time_since_update = 0  # reset because just updated
                trk.hits += 1
                if trk.state == TrackState.Tentative and trk.hits >= self.min_hits:
                    trk.state = TrackState.Confirmed

                # update orientation in propagated tracks and detected boxes so that they are within 90 degree
                # 根据检测到的目标框的朝向和跟踪器预测的朝向，通过orientation_correction函数来修正跟踪器预测的朝向，使其与检测到的目标框的朝向相近，
                # 从而使得跟踪器的预测更加准确。其中，bbox3d是检测到的目标框的3D坐标，trk.kf.x[3]是跟踪器预测的目标框的朝向。
                bbox3d = Box3D.bbox2array(dets[d[0]])  # x y z heading length width height
                trk.kf.x[3], bbox3d[3] = self.orientation_correction(trk.kf.x[3], bbox3d[3])

                # 更新轨迹的特征
                EMA_alpha = 0.9  # 刷新特征值占比
                feature = dets[d[0]].feature
                if feature is not None:
                    # 如果已有特征值，则刷新新特征如果没有特征值，则重新赋值
                    smooth_feat = EMA_alpha * trk.features[-1] + (1 - EMA_alpha) * feature if len(trk.features) else feature
                    trk.features = [smooth_feat]

                if trk.id == self.debug_id:
                    print('After ego-compoensation')
                    print(trk.kf.x.reshape((-1)))
                    print('matched measurement')
                    print(bbox3d.reshape((-1)))
                # print('uncertainty')
                # print(trk.kf.P)
                # print('measurement noise')
                # print(trk.kf.R)

                # kalman filter update with observation
                trk.kf.update(bbox3d)

                if trk.id == self.debug_id:
                    print('after matching')
                    print(trk.kf.x.reshape((-1)))
                    print('\n current velocity')
                    print(trk.get_velocity())

                trk.kf.x[3] = self.within_range(trk.kf.x[3])
                trk.info = info[d, :][0]

                # 更新最后一次成功匹配的2D位置
                center_point2d = np.array([(trk.info[2] + trk.info[4]) / 2, (trk.info[3] + trk.info[5]) / 2])
                trk.last_found = np.hstack((trk.info[2: -1], center_point2d))

            # 不管是否匹配上，轨迹的已存活时间加一
            trk.survival_time += 1
        # debug use only
        # else:
        # print('track ID %d is not matched' % trk.id)

    # 创新点: 生命周期管理之死亡
    def death(self, unmatched_trks, seq_name):
        if self.split == 'train' or self.split == 'val':
            # kitti数据集不同seq图像分辨率不一样
            if int(seq_name) in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
                width = 1242
                height = 375
            elif int(seq_name) in [14, 15, 16, 17]:
                width = 1224
                height = 370
            elif int(seq_name) in [18, 19]:
                width = 1238
                height = 374
            elif int(seq_name) == 20:
                width = 1241
                height = 376
        elif self.split == 'test':
            if int(seq_name) in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
                width = 1242
                height = 375
            elif int(seq_name) in [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]:
                width = 1224
                height = 370
            elif int(seq_name) in [27, 28]:
                width = 1226
                height = 370

        for track_idx in reversed(sorted(unmatched_trks)):
            # 离图中心的比例 中心出比例为0 边缘处比例为1
            width_per = abs(self.trackers[track_idx].last_found[-2] - width / 2) / (width / 2)
            height_per = abs(self.trackers[track_idx].last_found[-1] - height / 2) / (height / 2)
            length_per = (width_per + height_per)/2  # length_per = max(width_per, height_per)
            self.trackers[track_idx].death_possible = length_per * math.sqrt(self.trackers[track_idx].survival_time)

            # # 源代码
            # if self.trackers[track_idx].time_since_update >= self.max_age:
            #     self.trackers.pop(track_idx)

            # 改进1
            self.trackers[track_idx].mark_missed(self.unconfirm_max_age, self.confirm_max_age)
            self.trackers = [t for t in self.trackers if not t.is_deleted()]

            # 改进2：death, remove dead tracklet
            # 轨迹死亡从轨迹消失时间、轨迹存在时间、终点距离图像边缘距离三方面判定
            # if self.trackers[track_idx].death >= self.max_age:
            #     self.trackers.pop(track_idx)

    def birth(self, dets, info, unmatched_dets):
        # create and initialise new trackers for unmatched detections

        # dets = copy.copy(dets)
        new_id_list = list()  # new ID generated for unmatched detections
        for i in unmatched_dets:  # a scalar of index
            trk = KF(Box3D.bbox2array(dets[i]), dets[i].feature, info[i, :], self.ID_count[0])
            self.trackers.append(trk)
            new_id_list.append(trk.id)
            # print('track ID %s has been initialized due to new detection' % trk.id)

            self.ID_count[0] += 1

    # 改进后代码
    # 这段代码是一个跟踪器类的方法，用于输出已经稳定关联的物体的跟踪结果。
    def output(self):
        results = []
        for trk in self.trackers:
            # 只输出本帧匹配成功的结果 无论是不是confirmed track
            if trk.time_since_update == 0:
            # if ((trk.time_since_update < 2) and (
            #                          trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
                # change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
                d = Box3D.array2bbox(trk.kf.x[:7].reshape((7,)))  # bbox location self
                d = Box3D.bbox2array_raw(d)
                results.append(np.concatenate((d, [trk.id], trk.info)).reshape(1, -1))
        return results

    # # 源代码
    # def output(self):
    #     # output exiting tracks that have been stably associated, i.e., >= min_hits
    #     # and also delete tracks that have appeared for a long time, i.e., >= max_age
    #
    #     num_trks = len(self.trackers)
    #     results = []
    #     for trk in reversed(self.trackers):
    #         # change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
    #         d = Box3D.array2bbox(trk.kf.x[:7].reshape((7,)))  # bbox location self
    #         d = Box3D.bbox2array_raw(d)
    #
    #         if ((trk.time_since_update < self.max_age) and (
    #                 trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
    #             results.append(np.concatenate((d, [trk.id], trk.info)).reshape(1, -1))
    #         num_trks -= 1
    #
    #         # deadth, remove dead tracklet
    #         if (trk.time_since_update >= self.max_age):
    #             self.trackers.pop(num_trks)
    #
    #     return results

    # 这段代码实现了目标跟踪器（tracker）中的关键函数process_affi()，该函数用于处理目标检测结果与历史目标跟踪结果之间的关联矩阵，返回值是重新排序后的相似度矩阵。
    def process_affi(self, affi, matched, unmatched_dets, new_id_list):

        # post-processing affinity matrix, convert from affinity between raw detection and past total tracklets
        # to affinity between past "active" tracklets and current active output tracklets, so that we can know
        # how certain the results of matching is. The approach is to find the correspondes of ID for each row and
        # each column, map to the actual ID in the output trks, then purmute/expand the original affinity matrix

        ###### determine the ID for each past track
        trk_id = self.id_past  # ID in the trks for matching

        ###### determine the ID for each current detection
        det_id = [-1 for _ in range(affi.shape[0])]  # initialization

        # assign ID to each detection if it is matched to a track
        for match_tmp in matched:
            det_id[match_tmp[0]] = trk_id[match_tmp[1]]

        # assign the new birth ID to each unmatched detection
        count = 0
        assert len(unmatched_dets) == len(new_id_list), 'error'
        for unmatch_tmp in unmatched_dets:
            det_id[unmatch_tmp] = new_id_list[count]  # new_id_list is in the same order as unmatched_dets
            count += 1
        assert not (-1 in det_id), 'error, still have invalid ID in the detection list'

        ############################ update the affinity matrix based on the ID matching

        # transpose so that now row is past trks, col is current dets
        affi = affi.transpose()

        ###### compute the permutation for rows (past tracklets), possible to delete but not add new rows
        permute_row = list()
        for output_id_tmp in self.id_past_output:
            index = trk_id.index(output_id_tmp)
            permute_row.append(index)
        affi = affi[permute_row, :]
        assert affi.shape[0] == len(self.id_past_output), 'error'

        ###### compute the permutation for columns (current tracklets), possible to delete and add new rows
        # addition can be because some tracklets propagated from previous frames with no detection matched
        # so they are not contained in the original detection columns of affinity matrix, deletion can happen
        # because some detections are not matched

        max_index = affi.shape[1]
        permute_col = list()
        to_fill_col, to_fill_id = list(), list()  # append new columns at the end, also remember the ID for the added ones
        for output_id_tmp in self.id_now_output:
            try:
                index = det_id.index(output_id_tmp)
            except:  # some output ID does not exist in the detections but rather predicted by KF
                index = max_index
                max_index += 1
                to_fill_col.append(index)
                to_fill_id.append(output_id_tmp)
            permute_col.append(index)

        # expand the affinity matrix with newly added columns
        append = np.zeros((affi.shape[0], max_index - affi.shape[1]))
        append.fill(self.min_sim)
        affi = np.concatenate([affi, append], axis=1)

        # find out the correct permutation for the newly added columns of ID
        for count in range(len(to_fill_col)):
            fill_col = to_fill_col[count]
            fill_id = to_fill_id[count]
            row_index = self.id_past_output.index(fill_id)

            # construct one hot vector because it is proapgated from previous tracks, so 100% matching
            affi[row_index, fill_col] = self.max_sim
        affi = affi[:, permute_col]

        return affi

    def track(self, dets_all, frame, seq_name):
        """
		Params:
		  	dets_all: dict
				dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
				info: a array of other info for each det
			frame:    str, frame number, used to query ego pose
		Requires: this method must be called once for each frame even with empty detections.
		Returns the a similar array, where the last column is the object ID.

		NOTE: The number of objects returned may differ from the number of detections provided.
		"""
        dets, info = dets_all['dets'], dets_all['info']  # dets: N x 7, float numpy array
        if self.debug_id: print('\nframe is %s' % frame)

        # logging
        print_str = '\n\n*****************************************\n\nprocessing seq_name/frame %s/%d' % (
            seq_name, frame)
        print_log(print_str, log=self.log, display=False)
        self.frame_count += 1

        # recall the last frames of outputs for computing ID correspondences during affinity processing
        self.id_past_output = copy.copy(self.id_now_output)
        self.id_past = [trk.id for trk in self.trackers]

        # process detection format
        dets = self.process_dets(dets)

        # reid feature 打开图像耗时严重
        img_left_path = os.path.join(self.img_dir, f'{frame:06d}.png')
        img_right_path = os.path.join(self.img_right_dir, f'{frame:06d}.png')
        ori_left_img = cv2.imread(img_left_path)
        ori_right_img = cv2.imread(img_right_path)

        for det in dets:
            obj_get_feature(det, self.calib_reid, ori_left_img, ori_right_img, self.feature_extraction, self.cat)

        # tracks propagation based on velocity
        trks = self.prediction()

        # ego motion compensation, adapt to the current frame of camera coordinate
        if (frame > 0) and (self.ego_com) and (self.oxts is not None):
            trks = self.ego_motion_compensation(frame, trks)

        # visualization
        if self.vis and (self.vis_dir is not None):
            img = os.path.join(self.img_dir, f'{frame:06d}.png')
            save_path = os.path.join(self.vis_dir, f'{frame:06d}.jpg')
            mkdir_if_missing(save_path)
            self.visualization(img, dets, trks, self.calib, self.hw, save_path)

        # matching
        # dets：当前帧中检测到的所有目标检测框(det)，存储在一个列表中。
        # trks：上一帧中跟踪的所有目标跟踪器(track)，存储在一个列表中。
        # metric：距离度量方式
        # thres：阈值
        # algm：数据关联算法
        matched, unmatched_dets, unmatched_trks = data_association(dets, self.trackers, self.metric, self.thres, self.algm)
        # print_log('detections are', log=self.log, display=False)
        # print_log(dets, log=self.log, display=False)
        # print_log('tracklets are', log=self.log, display=False)
        # print_log(trks, log=self.log, display=False)
        # print_log('matched indexes are', log=self.log, display=False)
        # print_log(matched, log=self.log, display=False)
        # print_log('raw affinity matrix is', log=self.log, display=False)
        # print_log(affi, log=self.log, display=False)

        # update trks with matched detection measurement
        self.update(matched, unmatched_trks, dets, info)

        # create and initialise new trackers for unmatched detections
        self.birth(dets, info, unmatched_dets)

        # output existing valid tracks
        results = self.output()

        self.death(unmatched_trks, seq_name)

        if len(results) > 0:
            results = [np.concatenate(results)]  # h,w,l,x,y,z,theta, ID, other info, confidence
        else:
            results = [np.empty((0, 15))]
        self.id_now_output = results[0][:, 7].tolist()  # only the active tracks that are outputed

        # post-processing affinity to convert to the affinity between resulting tracklets
        # if self.affi_process:
        #     affi = self.process_affi(affi, matched, unmatched_dets, new_id_list)
        # print_log('processed affinity matrix is', log=self.log, display=False)
        # print_log(affi, log=self.log, display=False)

        # logging
        # print_log('\ntop-1 cost selected', log=self.log, display=False)
        # print_log(cost, log=self.log, display=False)
        for result_index in range(len(results)):
            print_log(results[result_index][:, :8], log=self.log, display=False)
            print_log('', log=self.log, display=False)

        return results
