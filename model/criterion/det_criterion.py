import torch
import torch.nn as nn
import torch.nn.functional as func


class DetCriterion(nn.Module):
    def __init__(self, no_instance_coef):
        super(DetCriterion, self).__init__()
        self.no_object_coef = no_instance_coef

    def forward(self, c, im, gc, gbm):
        cla_loss = self.get_label_loss(c, gc)
        mask_loss = self.get_mask_loss(im, gbm, gc)

        return cla_loss, mask_loss

    def get_label_loss(self, c, gc):
        num_classes = c.shape[-1]
        no_object_weight = torch.ones(num_classes).to(device=c.device)
        no_object_weight[0] = self.no_object_coef

        cla_loss = func.cross_entropy(input=c.transpose(dim0=1, dim1=2), target=gc.long(), weight=no_object_weight)

        return cla_loss

    @staticmethod
    def get_mask_loss(im, gbm, gc):
        instance_mask = gc.bool().float()  # [B, I]
        num_instances = instance_mask.sum(dim=-1)  # [B]
        # num_background = torch.sum(1. - gbm, dim=[-2, -1])  # [B, I]

        # mask_loss = func.l1_loss(im * (1. - gbm), torch.zeros_like(gbm), reduce=False)  # [B, I, H, W]
        mask_loss = func.l1_loss(func.sigmoid(im), gbm, reduce=False)
        mask_loss = mask_loss.mean(dim=[-2, -1]) * instance_mask

        mask_loss = mask_loss.sum(dim=-1) / (num_instances + 1e-7)
        mask_loss = mask_loss.mean()

        return mask_loss
