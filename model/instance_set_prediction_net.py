import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torchvision.models as cv_models

from .modules import StageBackbone, FeaturePyramidNetwork, InstanceClassModule, ClaBranch, BboxPredBranch


class InstanceSetPredictionNet(nn.Module):
    def __init__(self, num_classes, num_instances):
        super(InstanceSetPredictionNet, self).__init__()

        self.stage_backbone = StageBackbone(cv_models.resnet50(pretrained=True))
        self.fpn = FeaturePyramidNetwork(in_channels_list=[512, 1024, 2048], out_channels=2048)
        self.instance_class_module = InstanceClassModule(num_instances, num_classes, in_channels=2048)
        self.cla_branch = ClaBranch()
        self.bbox_pred_branch = BboxPredBranch(inner_dim=56)

    @amp.autocast()
    def forward(self, x):
        stage_feature_maps = self.stage_backbone(x)
        fpn_feature_maps = self.fpn(stage_feature_maps)

        instance_class_maps = [self.instance_class_module(feature_map) for feature_map in fpn_feature_maps]
        cla_logist = torch.stack([self.cla_branch(instance_class_map) for instance_class_map in instance_class_maps], dim=-1).mean(dim=-1)

        instance_map, pred_bbox = self.bbox_pred_branch(instance_class_maps, cla_logist)

        return cla_logist, pred_bbox, instance_map
