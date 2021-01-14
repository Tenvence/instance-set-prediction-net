import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func

from .det_criterion import DetCriterion
from .hungarian_matcher import HungarianMatcher


class Criterion(nn.Module):
    def __init__(self, class_weight, mask_weight, no_instance_coef, er_weight):
        super(Criterion, self).__init__()

        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.er_weight = er_weight

        self.matcher = HungarianMatcher(class_weight, mask_weight)
        self.det_criterion = DetCriterion(no_instance_coef)

    @staticmethod
    def _rearrange_idx(target, base_1, base_2):
        length = len(target)

        sim = np.expand_dims(base_1, axis=1).repeat(length, axis=1) == np.expand_dims(base_2, axis=0).repeat(length, axis=0)
        rearrange_target = np.expand_dims(target, axis=0).dot(sim).squeeze()

        return rearrange_target

    def _rearrange_instance_idx(self, pred_idx_1, gt_idx_1, pred_idx_2, gt_idx_2, num_instances):
        rearrange_pred_idx_1 = self._rearrange_idx(target=pred_idx_1, base_1=gt_idx_1, base_2=gt_idx_2)

        missed_pred_idx_1 = list(set(np.arange(num_instances)) - set(pred_idx_1))
        missed_pred_idx_2 = list(set(np.arange(num_instances)) - set(pred_idx_2))

        rearrange_pred_idx_1 = np.append(rearrange_pred_idx_1, missed_pred_idx_1)
        pred_idx_2 = np.append(pred_idx_2, missed_pred_idx_2)

        rearrange_pred_idx_1 = self._rearrange_idx(target=rearrange_pred_idx_1, base_1=pred_idx_2, base_2=np.arange(num_instances))

        return rearrange_pred_idx_1

    def _rearrange_instances_maps(self, pred_indices_1, gt_indices_1, pred_indices_2, gt_indices_2, im):
        batch_size, num_instances, _, _ = im.shape
        rearranged_im = torch.zeros_like(im)

        for batch_idx in range(batch_size):
            pred_idx_1, pred_idx_2 = pred_indices_1[batch_idx], pred_indices_2[batch_idx]
            gt_idx_1, gt_idx_2 = gt_indices_1[batch_idx], gt_indices_2[batch_idx]

            rearrange_pred_idx_1 = self._rearrange_instance_idx(pred_idx_1, gt_idx_1, pred_idx_2, gt_idx_2, num_instances)
            rearranged_im[batch_idx, np.arange(num_instances), :, :] = im.detach()[batch_idx, rearrange_pred_idx_1, :, :]

        return rearranged_im

    def _forward_set_criterion(self, cla_logist, bboxes_pred, classes_gt, bboxes_gt):
        match_classes_gt, match_bboxes_gt, pred_indices, gt_indices = self.matcher(cla_logist, bboxes_pred, classes_gt, bboxes_gt)
        cla_loss, mask_loss = self.det_criterion(cla_logist, bboxes_pred, match_classes_gt, match_bboxes_gt)

        return cla_loss, mask_loss, pred_indices, gt_indices

    def forward(self, c, aff_c, gc, im, aff_im, gbm):
        match_gc, match_gbm, pred_indices, gt_indices = self.matcher(c, im, gc, gbm)
        # aff_match_gc, aff_match_gbm, aff_pred_indices, aff_gt_indices = self.matcher(aff_c, aff_im, gc, gbm)

        # cla_loss, mask_loss = self.det_criterion(c, im, match_gc, match_gbm)
        cla_loss, mask_loss = self.det_criterion(c, im, match_gc, match_gbm)
        # aff_cla_loss, aff_mask_loss = self.det_criterion(aff_c, aff_im, aff_match_gc, aff_match_gbm)
        #
        # rearranged_im = self._rearrange_instances_maps(pred_indices, gt_indices, aff_pred_indices, aff_gt_indices, im)
        # rearranged_aff_im = self._rearrange_instances_maps(aff_pred_indices, aff_gt_indices, pred_indices, gt_indices, aff_im)
        # er_loss = func.l1_loss(aff_im, rearranged_im) + func.l1_loss(im, rearranged_aff_im)

        return cla_loss, mask_loss  # , aff_cla_loss, aff_mask_loss, er_loss
