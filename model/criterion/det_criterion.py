import torch
import torch.nn as nn
import torch.nn.functional as func

import utils.bbox_ops as bbox_ops


class DetCriterion(nn.Module):
    def __init__(self, no_instance_coef):
        super(DetCriterion, self).__init__()
        self.no_object_coef = no_instance_coef

    def forward(self, c, b, gc, gb):
        instance_mask = gc.bool().float()  # [B, I]
        num_instances = gc.bool().float().sum(dim=-1)  # [B]

        cla_loss = self.get_label_loss(c, gc)
        giou_loss = self.get_giou_loss(b, gb, instance_mask, num_instances)
        l1_loss = self.get_l1_loss(b, gb, instance_mask, num_instances)

        return cla_loss, giou_loss, l1_loss

    def get_label_loss(self, c, gc):
        num_classes = c.shape[-1]
        no_object_weight = torch.ones(num_classes).to(device=c.device)
        no_object_weight[0] = self.no_object_coef

        cla_loss = func.cross_entropy(input=c.transpose(dim0=1, dim1=2), target=gc.long(), weight=no_object_weight)

        return cla_loss

    @staticmethod
    def get_giou_loss(b, gb, instance_mask, num_instances):
        giou_loss = 1. - bbox_ops.get_pair_giou(bbox_ops.convert_bboxes_xywh_xyxy(b), bbox_ops.convert_bboxes_xywh_xyxy(gb))
        giou_loss *= instance_mask
        giou_loss = giou_loss.sum(dim=-1) / num_instances
        giou_loss = giou_loss.mean()

        return giou_loss

    @staticmethod
    def get_l1_loss(b, gb, instance_mask, num_instances):
        l1_loss = func.l1_loss(b, gb, reduce=False).mean(dim=-1) * instance_mask
        l1_loss = l1_loss.sum(dim=-1) / num_instances
        l1_loss = l1_loss.mean()

        return l1_loss
