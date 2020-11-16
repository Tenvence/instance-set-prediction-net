import torch
import torch.nn as nn
import torch.nn.functional as func
from scipy.optimize import linear_sum_assignment

import utils.bbox_ops as bbox_ops


class SetCriterion(nn.Module):
    def __init__(self, matcher, no_instance_coef, label_loss_coef, giou_loss_coef):
        super(SetCriterion, self).__init__()
        self.matcher = matcher
        self.no_object_coef = no_instance_coef
        self.label_loss_coef = label_loss_coef
        self.giou_loss_coef = giou_loss_coef

    def forward(self, cla_logist, bboxes_pred, classes_gt, bboxes_gt):
        matching_classes_gt, matching_bboxes_gt = self.matcher(cla_logist, bboxes_pred, classes_gt, bboxes_gt)  # pred_idx & gt_idx & real_object_mask: [B, num_queries]

        label_loss = self.get_label_loss(cla_logist, matching_classes_gt)
        giou_loss = self.get_bbox_loss(bboxes_pred, matching_bboxes_gt, matching_classes_gt)

        return self.label_loss_coef * label_loss + self.giou_loss_coef * giou_loss, label_loss, giou_loss

    def get_label_loss(self, logist_pred, classes_gt):
        num_classes = logist_pred.shape[-1]
        no_object_weight = torch.ones(num_classes).to(device=logist_pred.device)
        no_object_weight[0] = self.no_object_coef

        label_loss = func.cross_entropy(input=logist_pred.transpose(dim0=1, dim1=2), target=classes_gt.long(), weight=no_object_weight)

        return label_loss

    @staticmethod
    def get_bbox_loss(bboxes_pred, bboxes_gt, classes_gt):
        objects_mask = classes_gt.bool().float()  # [B, num_queries]
        num_instances = objects_mask.sum(dim=-1)  # [B]

        giou_loss = 1. - bbox_ops.get_pair_giou(bbox_ops.convert_bboxes_xywh_xyxy(bboxes_pred), bbox_ops.convert_bboxes_xywh_xyxy(bboxes_gt))
        giou_loss *= objects_mask
        giou_loss = giou_loss.sum(dim=-1) / (num_instances + 1e-7)
        giou_loss = giou_loss.mean()

        return giou_loss


class HungarianMatcher(nn.Module):
    def __init__(self, class_weight=1., giou_weight=1.):
        super(HungarianMatcher, self).__init__()
        self.class_weight = class_weight
        self.giou_weight = giou_weight

    @torch.no_grad()
    def forward(self, cla_logist, bboxes_pred, cla_gt, bboxes_gt):
        # cla_logist: [batch_size, num_queries, num_classes]
        # bboxes_pred: [batch_size, num_queries, 4]
        # classes_gt:  [batch_size, num_queries]
        # bboxes_gt:   [batch_size, num_queries, 4]

        batch_size, num_queries, num_classes = cla_logist.shape

        softmax_cla_logist = cla_logist.softmax(-1)
        classes_gt_one_hot = func.one_hot(cla_gt.long(), num_classes).float()  # [batch_size, num_queries, num_classes]

        # 1st dimension for batch size; 2nd dimension for prediction; 3rd dimension for ground-truth
        cost_label = -torch.bmm(softmax_cla_logist, classes_gt_one_hot.transpose(dim0=1, dim1=2))
        cost_giou = -bbox_ops.get_mutual_giou(bbox_ops.convert_bboxes_xywh_xyxy(bboxes_pred), bbox_ops.convert_bboxes_xywh_xyxy(bboxes_gt))

        matching_cost = self.class_weight * cost_label + self.giou_weight * cost_giou
        matching_cost = matching_cost.cpu().numpy()  # [B, num_queries, num_queries]

        real_object_numbers = cla_gt.bool().sum(dim=-1).cpu().numpy()  # store the real object numbers of the images in the batch

        matching_classes_gt = torch.zeros((batch_size, num_queries))  # same shape as "classes_gt"
        matching_bboxes_gt = torch.zeros((batch_size, num_queries, 4))  # same shape as "bboxes_gt"

        for i in range(batch_size):
            # "pred_idx" is the same shape as "gt_idx", whose length is "real_object_numbers[i]"
            pred_idx, gt_idx = linear_sum_assignment(matching_cost[i, :, :real_object_numbers[i]])
            pred_idx = torch.as_tensor(pred_idx).long()
            gt_idx = torch.as_tensor(gt_idx).long()

            matching_classes_gt[i, pred_idx] = cla_gt.cpu()[i, gt_idx]
            matching_bboxes_gt[i, pred_idx, :] = bboxes_gt.cpu()[i, gt_idx, :]

        return matching_classes_gt.long().to(device=cla_logist.device), matching_bboxes_gt.to(device=cla_logist.device)
