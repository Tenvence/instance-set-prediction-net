import torch
import torch.nn as nn
import torchvision.models as cv_models

from .modules import StageBackbone, FeaturePyramidNetwork, InstanceSeparator


class InstanceSetPredictionNet(nn.Module):
    def __init__(self, num_classes, num_instances, d_model):
        super(InstanceSetPredictionNet, self).__init__()

        self.stage_backbone = StageBackbone(cv_models.resnet50(pretrained=True, norm_layer=nn.SyncBatchNorm))
        self.fpn = FeaturePyramidNetwork(in_channels_list=[512, 1024, 2048], out_channels=d_model)

        self.instance_separator = InstanceSeparator(num_instances, num_classes, d_model)

    @staticmethod
    def _select(instance_class_map, cla_logist):
        _, _, _, h, w = instance_class_map.shape
        select_idx = torch.argmax(cla_logist, dim=-1, keepdim=True)[..., None, None].repeat(1, 1, 1, h, w)  # [B, I, 1] -> [B, I, 1, H, W]
        instance_map = torch.gather(instance_class_map, dim=2, index=select_idx).squeeze()  # [B, I, C, H, W] -> [B, I, 1, H, W] -> [B, I, H, W]
        return instance_map

    def forward(self, x):
        _, _, h, w = x.shape

        c1, c2, c3, c4, c5 = self.stage_backbone(x)
        fpn_feature_map = self.fpn([c3, c4, c5])

        instance_class_map = self.instance_separator(fpn_feature_map)
        instance_class_vec = torch.mean(instance_class_map, dim=[-1, -2], keepdim=True)  # [B, I, C, 1, 1]

        cla_logist = instance_class_vec.squeeze()  # [B, I, C]
        instance_map = torch.sum(instance_class_map * torch.softmax(instance_class_vec, dim=2), dim=2)  # [B, I, H, W]

        return cla_logist, instance_map
