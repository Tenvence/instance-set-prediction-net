import torch
import torch.nn as nn
import torch.nn.functional as func
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    def __init__(self, class_weight, mask_weight):
        super(HungarianMatcher, self).__init__()
        self.class_weight = class_weight
        self.mask_weight = mask_weight

    @staticmethod
    def _get_bbox_mask_cost(im, gbm):
        num_instances = im.shape[1]
        repeated_im = im[:, :, None, :, :].repeat(1, 1, num_instances, 1, 1)  # [B, I, I, H, W]
        repeated_gbm = gbm[:, None, :, :, :].repeat(1, num_instances, 1, 1, 1)  # [B, I, I, H, W]

        # cost = func.binary_cross_entropy_with_logits(repeated_im, repeated_gbm, reduce=False).mean(dim=[-2, -1])
        cost = func.l1_loss(func.sigmoid(repeated_im), repeated_gbm, reduce=False).mean(dim=[-2, -1])  # [B, I, I, H, W] -> [B, I, I]

        return cost

    @torch.no_grad()
    def forward(self, c, im, gc, gbm):
        batch_size, num_queries, num_classes = c.shape
        _, _, h, w = im.shape

        softmax_cla_logist = c.softmax(-1)
        classes_gt_one_hot = func.one_hot(gc.long(), num_classes).float()  # [batch_size, num_queries, num_classes]

        # 1st dimension for batch size; 2nd dimension for prediction; 3rd dimension for ground-truth
        cost_label = -torch.bmm(softmax_cla_logist, classes_gt_one_hot.transpose(dim0=1, dim1=2))
        cost_bbox_mask = self._get_bbox_mask_cost(im, gbm)

        matching_cost = self.class_weight * cost_label + self.mask_weight * cost_bbox_mask
        matching_cost = matching_cost.cpu().numpy()  # [B, num_queries, num_queries]

        real_instance_numbers = gc.bool().sum(dim=-1).cpu().numpy()  # store the real object numbers of the images in the batch

        matching_gc = torch.zeros((batch_size, num_queries)).cuda()  # same shape as "classes_gt"
        matching_gbm = torch.zeros((batch_size, num_queries, h, w)).cuda()  # same shape as "bboxes_gt"

        pred_indices = []
        gt_indices = []
        for i in range(batch_size):
            # "pred_idx" is the same shape as "gt_idx", whose length is "real_object_numbers[i]"
            pred_idx, gt_idx = linear_sum_assignment(matching_cost[i, :, :real_instance_numbers[i]])

            pred_indices.append(pred_idx)
            gt_indices.append(gt_idx)

            pred_idx = torch.as_tensor(pred_idx).long()
            gt_idx = torch.as_tensor(gt_idx).long()

            matching_gc[i, pred_idx] = gc[i, gt_idx]
            matching_gbm[i, pred_idx, ...] = gbm[i, gt_idx, ...]

        return matching_gc.long(), matching_gbm, pred_indices, gt_indices
