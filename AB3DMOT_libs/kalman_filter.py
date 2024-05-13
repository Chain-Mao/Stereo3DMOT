import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Filter(object):
    def __init__(self, bbox3D, feature, info, ID):
        self.initial_pos = bbox3D
        self.time_since_update = 0
        self.id = ID
        self.hits = 1  # number of total hits including the first detection
        self.info = info  # other information associated
        self.last_found = np.zeros(6)  # 最后一次跟踪到的坐标
        self.survival_time = 0  # 目标已经存活的时间
        self.death_possible = 0  # 目标死亡累积阈值
        self.state = TrackState.Tentative  # 状态

        self.features = []  # 轨迹特征集合
        self.box = None

        if feature is not None:
            self.features.append(feature)  # 如果存在初始特征，则添加


class KF(Filter):
    def __init__(self, bbox3D, feature, info, ID):
        super().__init__(bbox3D, feature, info, ID)

        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        # There is no need to use EKF here as the measurement and state are in the same space with linear relationship

        # state x dimension 10: x, y, z, theta, l, w, h, dx, dy, dz
        # constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz
        # while all others (theta, l, w, h, dx, dy, dz) remain the same
        self.kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # state transition matrix, dim_x * dim_x
                              [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        # measurement function, dim_z * dim_x, the first 7 dimensions of the measurement correspond to the state
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

        # measurement uncertainty, uncomment if not super trust the measurement data due to detection noise
        # self.kf.R[0:,0:] *= 10.

        # initial state uncertainty at time 0
        # Given a single data, the initial velocity is very uncertain, so giv a high uncertainty to start
        self.kf.P[7:, 7:] *= 1000.
        self.kf.P *= 10.

        # process uncertainty, make the constant velocity part more certain
        self.kf.Q[7:, 7:] *= 0.01

        # initialize data
        self.kf.x[:7] = self.initial_pos.reshape((7, 1))

    def compute_innovation_matrix(self):
        """ compute the innovation matrix for association with mahalanobis distance
		"""
        return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R

    def get_velocity(self):
        # return the object velocity in the state

        return self.kf.x[7:]

    def mark_missed(self, unconfirm_max_age, confirm_max_age):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            if self.time_since_update > unconfirm_max_age:
                self.state = TrackState.Deleted
        elif self.time_since_update + self.death_possible > confirm_max_age:
            self.state = TrackState.Deleted

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted