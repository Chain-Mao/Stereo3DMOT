import copy
import os

import cv2
import numpy as np
from PIL import Image
from numba import jit
from scipy.optimize import linear_sum_assignment

from AB3DMOT_libs import linear_assignment
from AB3DMOT_libs.box import Box3D, read_obj_data
from AB3DMOT_libs.dist_metrics import iou, dist3d, dist_ground, m_distance, feature_similarity
from AB3DMOT_libs.kalman_filter import TrackState
from AB3DMOT_libs.linear_assignment import gated_metric


# 源代码
def compute_affinity(dets, trks, metric):
    # compute affinity matrix
    aff_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):

            # choose to use different distance metrics
            if 'iou' in metric:
                dist_now = iou(det, trk.box, metric)
            else:
                assert False, 'error'
            aff_matrix[d, t] = dist_now

    return aff_matrix


def compute_affinity_round1(dets, tracks, round1_detections, round1_tracks, metric, iou_threshold):
    assert 'iou' in metric, "本算法只支持iou运算"
    # compute affinity matrix
    # 计算检测框（dets）和跟踪框（trks）之间的相似度（affinity）矩阵。
    aff_matrix = np.full((len(dets), len(tracks)), -1000, dtype=np.float32)
    feature_ratio = 0.9  # 特征占比 0.9
    iou_ratio = 1 - feature_ratio  # iou占比

    # 第一轮匹配有特征值的detections和tracks
    for d in round1_detections:
        for t in round1_tracks:
            aff_matrix[d, t] = feature_similarity(dets[d].feature, tracks[t].features[-1]) * feature_ratio + iou(
                dets[d], tracks[t].box, metric) * iou_ratio
            # 去除特征相似度和IOU交并比低于阈值的detections和tracks
            if iou(dets[d], tracks[t].box, metric) < iou_threshold:
                # iou过低则放弃
                aff_matrix[d, t] = -1000
    return aff_matrix  # 大小为(len(dets), len(trks))


def compute_affinity_round2(dets, tracks, round2_detections, round2_tracks, metric):
    assert 'iou' in metric, "本算法只支持iou运算"
    # compute affinity matrix
    # 计算检测框（dets）和跟踪框（trks）之间的相似度（affinity）矩阵。
    aff_matrix = np.full((len(dets), len(tracks)), -1000, dtype=np.float32)

    # 第二轮匹配剩余的detections和tracks
    for d in round2_detections:
        for t in round2_tracks:
            aff_matrix[d, t] = iou(dets[d], tracks[t].box, metric)
    return aff_matrix  # 大小为(len(dets), len(trks))


def greedy_matching(cost_matrix):
    # association in the greedy manner
    # refer to https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/blob/master/main.py

    num_dets, num_trks = cost_matrix.shape[0], cost_matrix.shape[1]

    # sort all costs and then convert to 2D
    distance_1d = cost_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_trks, index_1d % num_trks], axis=1)

    # assign matches one by one given the sorting, but first come first serves
    det_matches_to_trk = [-1] * num_dets
    trk_matches_to_det = [-1] * num_trks
    matched_indices = []
    for sort_i in range(index_2d.shape[0]):
        det_id = int(index_2d[sort_i][0])
        trk_id = int(index_2d[sort_i][1])

        # if both id has not been matched yet
        if trk_matches_to_det[trk_id] == -1 and det_matches_to_trk[det_id] == -1:
            trk_matches_to_det[trk_id] = det_id
            det_matches_to_trk[det_id] = trk_id
            matched_indices.append([det_id, trk_id])

    return np.asarray(matched_indices)


def obj_get_feature(obj, calib_reid, ori_left_img, ori_right_img, feature_extraction, det_cls):
    object = read_obj_data(obj, calib=calib_reid, im_shape=ori_left_img.shape)
    # 大部分图像在左右目中均有展现，双目深度reid只针对这些图像，其他异常图像退化为仅用运动特征跟踪
    y_min = int(min(object.boxes[0].box[1], object.boxes[1].box[1]))
    y_max = int(max(object.boxes[0].box[3], object.boxes[1].box[3]))

    # 固定x框的长度
    x_length_max = int(
        max(object.boxes[0].box[2] - object.boxes[0].box[0], object.boxes[1].box[2] - object.boxes[1].box[0]))
    x_length_min = int(
        min(object.boxes[0].box[2] - object.boxes[0].box[0], object.boxes[1].box[2] - object.boxes[1].box[0]))
    xcam2_center = int((object.boxes[0].box[0] + object.boxes[0].box[2]) / 2)
    xcam3_center = int((object.boxes[1].box[0] + object.boxes[1].box[2]) / 2)
    if xcam2_center - int(x_length_max / 2) > 0 and xcam3_center - int(x_length_max / 2) > 0 and xcam2_center + int(x_length_max / 2) < ori_left_img.shape[1] and xcam3_center + int(x_length_max / 2) < ori_left_img.shape[1]:
        x_length = x_length_max
    else:
        x_length = x_length_min

    # 处理10像素以上的图像
    if object.boxes[0].box[2] > object.boxes[0].box[0] + 10 and object.boxes[1].box[2] > object.boxes[1].box[0] + 10:
        crop_left = ori_left_img[y_min: y_max, xcam2_center - int(x_length / 2): xcam2_center + int(x_length / 2)]
        crop_right = ori_right_img[y_min: y_max, xcam3_center - int(x_length / 2): xcam3_center + int(x_length / 2)]
        assert crop_left.shape == crop_right.shape, "左右图shape不一致"
        # 左右crop图偏移量统计
        obj.disp_offset = object.boxes[0].box[0] - object.boxes[1].box[0]
        obj.hw = crop_left.shape[0:2]  # 前高后宽
        # 收集obj的特征向量
        obj.feature, obj.disp = feature_extraction.run_on_image(crop_left, crop_right, det_cls, obj.disp_offset, xcam2_center - int(x_length / 2), y_min, calib_reid.p2[0,0], calib_reid.p2[0,2], calib_reid.p2[1,2])
        # cv2.imshow("crop_left", crop_left)
        # cv2.imshow("crop_right", crop_right)
        # cv2.waitKey(0)
    else:
        obj.feature, obj.disp = None, None

    # # 转到cpu
    # if obj.feature is not None and obj.disp is not None:
    #     obj.feature = obj.feature.cpu()
    #     obj.disp = obj.disp.cpu()


def match_analyse(aff_matrix, algm, detections_number, tracks_number, threshold):
    # 如果detections或者tracks为空，则不匹配
    if len(detections_number) == 0:
        return np.empty((0, 2), dtype=int), [], tracks_number
    if len(tracks_number) == 0:
        return np.empty((0, 2), dtype=int), detections_number, []

    # association based on the affinity matrix
    if algm == 'hungar':
        row_ind, col_ind = linear_sum_assignment(-aff_matrix)  # hougarian algorithm
        matched_indices_all = np.stack((row_ind, col_ind), axis=1)
    elif algm == 'greedy':
        matched_indices_all = greedy_matching(-aff_matrix)  # greedy matching
    else:
        assert False, 'error'

    # 找出在待匹配列表中成功匹配的detections和tracks，取交集保证仅关注待匹配列表中的数据
    matched_indices = np.empty((0, 2), dtype=int)
    for matched_indice in matched_indices_all:
        if matched_indice[0] in detections_number and matched_indice[1] in tracks_number:
            matched_indices = np.row_stack((matched_indices, matched_indice))

    # save for unmatched objects
    unmatched_dets = []
    for d in detections_number:
        if d not in matched_indices[:, 0]: unmatched_dets.append(d)
    unmatched_trks = []
    for t in tracks_number:
        if t not in matched_indices[:, 1]: unmatched_trks.append(t)

    # filter out matches with low affinity
    matches = []
    for m in matched_indices:
        if aff_matrix[m[0], m[1]] < threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, unmatched_dets, unmatched_trks


def data_association(dets, tracks, metric, threshold, algm='greedy'):
    """
    Assigns detections to tracked object

    dets:  a list of Box3D object
    trks:  a list of Box3D object

    Returns 3 lists of matches, unmatched_dets and unmatched_trks, and total cost, and affinity matrix
    """

    # if there is no item in either row/col, skip the association and return all as unmatched
    if len(tracks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(dets)), []
    if len(dets) == 0:
        return np.empty((0, 2), dtype=int), [], np.arange(len(tracks))

    # # 源代码
    # # compute affinity matrix
    # aff_matrix = compute_affinity(dets, tracks, metric)
    #
    # # association based on the affinity matrix
    # if algm == 'hungar':
    #     row_ind, col_ind = linear_sum_assignment(-aff_matrix)  # hougarian algorithm
    #     matched_indices = np.stack((row_ind, col_ind), axis=1)
    # elif algm == 'greedy':
    #     matched_indices = greedy_matching(-aff_matrix)  # greedy matching
    # else:
    #     assert False, 'error'
    #
    # # compute total cost
    # cost = 0
    # for row_index in range(matched_indices.shape[0]):
    #     cost -= aff_matrix[matched_indices[row_index, 0], matched_indices[row_index, 1]]
    #
    # # save for unmatched objects
    # unmatched_dets = []
    # for d, det in enumerate(dets):
    #     if (d not in matched_indices[:, 0]): unmatched_dets.append(d)
    # unmatched_trks = []
    # for t, trk in enumerate(tracks):
    #     if (t not in matched_indices[:, 1]): unmatched_trks.append(t)
    #
    # # filter out matches with low affinity
    # matches = []
    # for m in matched_indices:
    #     if (aff_matrix[m[0], m[1]] < threshold):
    #         unmatched_dets.append(m[0])
    #         unmatched_trks.append(m[1])
    #     else:
    #         matches.append(m.reshape(1, 2))
    # if len(matches) == 0:
    #     matches = np.empty((0, 2), dtype=int)
    # else:
    #     matches = np.concatenate(matches, axis=0)
    #
    # return matches, np.array(unmatched_dets), np.array(unmatched_trks)

    # 自行设计
    # 将检测目标分为特征的和无特征的两部分，分别存储在feature_detections 和 nofeature_detections 列表中。
    round1_detections = [
        i for i, d in enumerate(dets) if d.feature is not None]
    round2_detections = [
        i for i, d in enumerate(dets) if d.feature is None]

    round1_tracks = [
        i for i, t in enumerate(tracks) if len(t.features)]
    round2_tracks = [
        i for i, t in enumerate(tracks) if not len(t.features)]

    aff_matrix_round1 = compute_affinity_round1(dets, tracks, round1_detections, round1_tracks, metric, iou_threshold=threshold)

    # threshold为特征值阈值
    matches_round1, unmatched_dets_round1, unmatched_trks_round1 = match_analyse(aff_matrix_round1, algm, round1_detections, round1_tracks, threshold=0.6)

    # 第一梯队剩余detections和tracks加入第二梯队
    round2_detections = unmatched_dets_round1 + round2_detections
    round2_tracks = unmatched_trks_round1 + round2_tracks

    aff_matrix_round2 = compute_affinity_round2(dets, tracks, round2_detections, round2_tracks, metric)
    matches_round2, unmatched_dets_round2, unmatched_trks_round2 = match_analyse(aff_matrix_round2, algm, round2_detections, round2_tracks, threshold=threshold)

    matches = np.append(matches_round1, matches_round2, axis=0)

    return matches, np.array(unmatched_dets_round2), np.array(unmatched_trks_round2)

    #
    # # 参考StrongSORT
    # # 第一轮中的目标需结合外观特征和运动信息，第二轮中的目标仅有运动信息
    # # AB3DMOT中compute affinity matrix 用IOU度量，trk_inv_inn_matrices为None
    # '''
    # 计划两次匹配
    # 第一次：IOU+Reid+Depth匹配可以参加特征提取和深度匹配的目标
    # 滤除置信度过低结果
    # 第二次：IOU匹配剩余不可参加的目标
    # 滤除置信度过低结果
    # '''
    # # 将检测目标分为特征的和无特征的两部分，分别存储在feature_detections 和 nofeature_detections 列表中。
    # feature_detections = [
    #     i for i, d in enumerate(dets) if d.feature is not None]
    # nofeature_detections = [
    #     i for i, d in enumerate(dets) if d.feature is None]
    #
    # # Split track set into confirmed and unconfirmed tracks.
    # # 确认态并且存在feature
    # confirmed_feature_tracks = [
    #     i for i, t in enumerate(tracks) if len(t.features) and t.is_confirmed()]
    # # 不确认态或者不存在feature
    # unconfirmed_feature_tracks = [
    #     i for i, t in enumerate(tracks) if not len(t.features) or not t.is_confirmed()]
    #
    # matching_threshold = 0.35
    # # Associate confirmed tracks using appearance features.
    # matches_a, unmatched_tracks_a, unmatched_detections = \
    #     linear_assignment.matching_cascade(
    #         gated_metric, matching_threshold,
    #         tracks, dets, confirmed_feature_tracks, feature_detections)
    #
    # # Associate remaining tracks together with unconfirmed tracks using IOU.
    # iou_track_candidates = unconfirmed_feature_tracks + [
    #     k for k in unmatched_tracks_a if
    #     tracks[k].time_since_update == 1]
    # unmatched_tracks_a = [
    #     k for k in unmatched_tracks_a if
    #     tracks[k].time_since_update != 1]
    #
    # iou_det_candidates = unmatched_detections + nofeature_detections
    #
    # # IOU阈值
    # max_iou_distance = 0.7
    # matches_b, unmatched_tracks_b, unmatched_detections = \
    #     linear_assignment.min_cost_matching(
    #         linear_assignment.iou_distance, max_iou_distance, tracks,
    #         dets, iou_track_candidates, iou_det_candidates)
    #
    # matches = matches_a + matches_b
    # unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
    #
    # return np.asarray(matches), np.array(unmatched_detections), np.array(unmatched_tracks)
    #
