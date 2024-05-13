# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
import torch
from sklearn.utils.linear_assignment_ import linear_assignment
from AB3DMOT_libs.dist_metrics import iou

INFTY_COST = 1e+5


def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    cost_matrix_ = cost_matrix.copy()

    indices = linear_assignment(cost_matrix_)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            # matches.append((track_idx, detection_idx))
            matches.append((detection_idx, track_idx))  # 符合AB3DMOT中写法
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
        distance_metric, max_distance, tracks, detections,
        track_indices=None, detection_indices=None):
    """Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    track_indices_l = [k for k in track_indices]
    matches_l, _, unmatched_detections = \
        min_cost_matching(
            distance_metric, max_distance, tracks, detections,
            track_indices_l, unmatched_detections)
    matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for _, k in matches))
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(feat_dis_matrix, tracks, detections, track_indices, detection_indices):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)), dtype=np.float32)
    iou_dis_threshold = 5
    feature_lambda = 0.9
    for row, track_idx in enumerate(track_indices):
        for col, det_idx in enumerate(detection_indices):
            iou_dis = 1 - iou(tracks[row].box, detections[col], 'iou_3d')
            cost_matrix[row, col] = feature_lambda * feat_dis_matrix[row, col] + (1 - feature_lambda) * iou_dis
            cost_matrix[row, col] = INFTY_COST if iou_dis > iou_dis_threshold else cost_matrix[row, col]
    return cost_matrix


def gated_metric(tracks, detections, track_indices, detection_indices):
    detection_features = list([detections[i].feature for i in detection_indices])
    track_features = list([tracks[i].features for i in track_indices])
    feat_dis_matrix = feature_distance(detection_features, track_features)
    cost_matrix = gate_cost_matrix(
        feat_dis_matrix, tracks, detections, track_indices,
        detection_indices)

    return cost_matrix


def feature_distance(detection_features, track_features):
    """Compute distance between features and targets.

    Parameters
    ----------
    features : ndarray
        An NxM matrix of N features of dimensionality M.
    targets : List[int]
        A list of targets to match the given `features` against.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape len(targets), len(features), where
        element (i, j) contains the closest squared distance between
        `targets[i]` and `features[j]`.

    """
    feat_dis_matrix = np.zeros((len(track_features), len(detection_features)))
#     # 计算检测框（dets）和跟踪框（trks）之间的相似度（affinity）矩阵。其中，距离度量方式可以是iou、m_dis、euler、dist_2d、dist_3d中的一种。
    for t, trk_feature in enumerate(track_features):
        for d, det_feature in enumerate(detection_features):
            feat_dis_matrix[t, d] = 1 - torch.cosine_similarity(trk_feature[0], det_feature)

    return feat_dis_matrix  # 大小为(len(trks), len(dets))


def iou_distance(tracks, detections, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        for col, det_idx in enumerate(detection_indices):
            cost_matrix[row][col] = 1 - iou(tracks[track_idx].box, detections[det_idx], 'iou_3d')
            # 剔除time_since_update过大值，有待考虑
            if tracks[track_idx].time_since_update > 1:
                cost_matrix[row, col] = INFTY_COST
    return cost_matrix