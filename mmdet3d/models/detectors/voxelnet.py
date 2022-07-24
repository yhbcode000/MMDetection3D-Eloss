# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import Voxelization
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from .. import builder
from ..builder import DETECTORS
from .single_stage import SingleStage3DDetector


@DETECTORS.register_module()
class VoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 
                 with_eloss=False,
                 
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(VoxelNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)
        self.with_eloss = with_eloss
        if with_eloss:
            self.eloss = builder.build_loss(dict(type = 'EntropyLoss'))
            
        # reference: https://blog.csdn.net/qq_21997625/article/details/90369838
        for i,p in enumerate(self.parameters()):
            # exp02-eloss33
            if i > 23:
                p.requires_grad = False
                
            # exp02-eloss66
            # if i < 24 and i > 59:
            #     p.requires_grad = False
            
            # exp02-eloss99
            # if i < 60 and i > 95:
            #     p.requires_grad = False
        
    def add_gauss_noise(self, points, prob=1.0, level=1):
        for i in range(len(points)):
            if not i % (1//prob) != 0:
                for _ in range(level):
                    points[i] += torch.randn_like(points[i])
        return points
    
    def add_sp_noise(self, points, prob=0.5):  
        max_point = torch.max(points)
        min_point = torch.min(points)
        
        for i in range(len(points)):
            if not i % (1//prob) != 0:
                if not i % 2 != 0:
                    points[i] = max_point
                else:
                    points[i] = min_point
                    
        return points
    
    def extract_feat(self, points, img_metas=None):
        """Extract features from points."""
        
        points = self.add_gauss_noise(points, prob=0.3, level=1)
        
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = torch.as_tensor(coors[-1, 0].item() + 1)
        x = self.middle_encoder(voxel_features, coors, batch_size)
        
        if self.with_eloss:
            x, net_info = self.backbone(x)
            if self.with_neck:
                x = self.neck(x)
            return x, net_info
        else:
            x = self.backbone(x)
            if self.with_neck:
                x = self.neck(x)
            return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        if self.with_eloss:
            x, net_info = self.extract_feat(points, img_metas)
        else:
            x = self.extract_feat(points, img_metas)
            
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        
        if self.with_eloss:
            losses.update(self.eloss(net_info))
            
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        if self.with_eloss:
            x, net_info = self.extract_feat(points, img_metas)
        else:
            x = self.extract_feat(points, img_metas)
            
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]
