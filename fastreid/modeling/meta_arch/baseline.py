# encoding: utf-8
import cv2
import torch
from torch import nn
import numpy as np
from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
from Fast_ACVNet.Fast_ACV_plus import Fast_ACVNet_plus_head
from Fast_ACVNet.loss import model_loss_train
from pointnet.model import PointNetfeat
from fastreid.data.common import scale_nonzero_values, depth_to_point_cloud


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            heads,
            pixel_mean,
            pixel_std,
            loss_kwargs=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone head组成
        # backbone
        self.backbone = backbone
        self.point_backbone = PointNetfeat()

        # head
        self.heads = heads
        self.disp_head = Fast_ACVNet_plus_head(192, False)

        # loss head
        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    # 建立backbone和和heads
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)
        return {
            'backbone': backbone,
            'heads': heads,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    }
                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    # 整个神经网络forward代码
    # 输入图像，经过组装的神经网络，进行训练/推理
    # 在分别训练车、行人、自行车时，输入网络的width和height不一样
    def forward(self, batched_inputs):
        left_images, right_images = self.preprocess_image(batched_inputs)
        left_features, left_disp_features = self.backbone(left_images)
        right_features, right_disp_features = self.backbone(right_images)

        RGB_features = torch.cat([left_features, right_features], dim=1)

        # 计算实例视差
        object_disp_ests = self.disp_head(left_images, right_images, left_disp_features, right_disp_features)

        # 余弦相似度计算两个特征相似程度
        # 将高维向量展平成一维向量
        vector1 = left_features.view(left_features.size(0), -1)
        vector2 = right_features.view(right_features.size(0), -1)
        # 计算余弦相似度,相似度过低则从特征匹配列表转到非特征匹配列表
        similarity = torch.cosine_similarity(vector1, vector2)

        if self.training:  # 训练
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            ''' 正常训练'''
            # 训练时用真值视差产生的点云进行特征提取
            point_features = self.point_backbone(batched_inputs['xyz_sampled'])
            outputs = self.heads(RGB_features, point_features.unsqueeze(2).unsqueeze(3), targets)
            losses = self.losses(outputs, targets)

            # disp loss computer
            disp_gt, disp_gt_low, disp_offset = batched_inputs['disp_images'], batched_inputs['disp_low_images'], batched_inputs['disp_offset']
            disp_offset = disp_offset.unsqueeze(1).unsqueeze(2)  # reshape to (32,1,1)
            disp_ests = [object_disp_est + disp_offset for object_disp_est in object_disp_ests]
            mask = (disp_gt < 192) & (disp_gt > 0)
            mask_low = (disp_gt_low < 192) & (disp_gt_low > 0)
            masks = [mask, mask_low, mask, mask_low]
            disp_gts = [disp_gt, disp_gt_low, disp_gt, disp_gt_low]
            disp_loss = model_loss_train(disp_ests, disp_gts, masks)
            # 使得各个loss在统一量级
            losses["loss_disp"] = disp_loss/30

            # ''' 微调 '''
            # point_features = self.point_backbone(batched_inputs['xyz_sampled'])
            # outputs = self.heads(RGB_features, point_features.unsqueeze(2).unsqueeze(3), targets)
            # losses = self.losses(outputs, targets)
            return losses

        else:  # 推理或评估
            if len(similarity) == 1:  # 推理
                # 如果余弦相似度过低，放弃输出特征
                if similarity < 0.5:
                    return None, None
                else:
                    object_disp = object_disp_ests[-1]
                    disp_est = cv2.resize((object_disp + batched_inputs['disp_offset']).cpu().numpy().squeeze(), (batched_inputs['ori_hw'][1], batched_inputs['ori_hw'][0]), interpolation=cv2.INTER_NEAREST)
                    # 得到原始深度图
                    mask = (disp_est != 0)
                    depth_img = np.zeros_like(disp_est, dtype=np.float32)
                    depth_img[mask] = 1 / disp_est[mask]
                    depth_image = scale_nonzero_values(depth_img)
                    xyz_sampled = depth_to_point_cloud(depth_image, batched_inputs['x_min'], batched_inputs['y_min'], batched_inputs['focal'], batched_inputs['cx'], batched_inputs['cy'])
                    point_features = self.point_backbone(xyz_sampled.unsqueeze(0).cuda())
                    outputs = self.heads(RGB_features, point_features.unsqueeze(2).unsqueeze(3))
                    return outputs, disp_est
            else:   # 评估 只评估reid部分
                point_features = self.point_backbone(batched_inputs['xyz_sampled'])
                outputs = self.heads(RGB_features, point_features.unsqueeze(2).unsqueeze(3))
                return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            left_images = batched_inputs['left_images']
            right_images = batched_inputs['right_images']
        # elif isinstance(batched_inputs, torch.Tensor):
        #     images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        left_images.sub_(self.pixel_mean).div_(self.pixel_std)
        right_images.sub_(self.pixel_mean).div_(self.pixel_std)
        return left_images, right_images

    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs = outputs['cls_outputs']
        pred_features = outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        # 损失函数
        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')

        return loss_dict