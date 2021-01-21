import torch
import torch.nn as nn
import torch.nn.functional as func
from scipy.optimize import linear_sum_assignment

import utils.bbox_ops as bbox_ops


class HungarianMatcher(nn.Module):
    def __init__(self, class_weight, giou_weight, l1_weight):
        super(HungarianMatcher, self).__init__()
        self.class_weight = class_weight
        self.giou_weight = giou_weight
        self.l1_weight = l1_weight

    @torch.no_grad()
    def forward(self, c, b, gc, gb):
        batch_size, num_instances, num_classes = c.shape

        softmax_cla_logist = c.softmax(-1)
        classes_gt_one_hot = func.one_hot(gc.long(), num_classes).float()  # [batch_size, num_queries, num_classes]

        # 1st dimension for batch size; 2nd dimension for prediction; 3rd dimension for ground-truth
        cost_cla = -torch.bmm(softmax_cla_logist, classes_gt_one_hot.transpose(dim0=1, dim1=2))
        cost_giou = -bbox_ops.get_mutual_giou(bbox_ops.convert_bboxes_xywh_xyxy(b), bbox_ops.convert_bboxes_xywh_xyxy(gb))
        cost_l1 = torch.cdist(b, gb, p=1)

        matching_cost = self.class_weight * cost_cla + self.giou_weight * cost_giou + self.l1_weight * cost_l1
        matching_cost = matching_cost.cpu().numpy()  # [B, num_queries, num_queries]

        real_object_numbers = gc.bool().sum(dim=-1).cpu().numpy()  # store the real object numbers of the images in the batch

        match_gc = torch.zeros((batch_size, num_instances)).cuda()  # same shape as "classes_gt"
        match_gb = torch.zeros((batch_size, num_instances, 4)).cuda()  # same shape as "bboxes_gt"

        for i in range(batch_size):
            # "pred_idx" is the same shape as "gt_idx", whose length is "real_object_numbers[i]"
            pred_idx, gt_idx = linear_sum_assignment(matching_cost[i, :, :real_object_numbers[i]])
            pred_idx = torch.as_tensor(pred_idx).long()
            gt_idx = torch.as_tensor(gt_idx).long()

            match_gc[i, pred_idx] = gc[i, gt_idx]
            match_gb[i, pred_idx, :] = gb[i, gt_idx, :]

        return match_gc.long(), match_gb
