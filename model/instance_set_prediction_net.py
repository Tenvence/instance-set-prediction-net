import torch.nn as nn
import torchvision.models as cv_models

from .modules import InstanceClassModule, ClaBranch, BboxPredBranch


class InstanceSetPredictionNet(nn.Module):
    def __init__(self, num_classes, num_instances):
        super(InstanceSetPredictionNet, self).__init__()

        self.backbone = nn.Sequential(*list(cv_models.resnet50(pretrained=True).children())[:-2])
        self.instance_class_module = InstanceClassModule(num_instances, num_classes, in_channels=2048)
        self.cla_branch = ClaBranch()
        self.bbox_pred_branch = BboxPredBranch(inner_dim=14)

    def forward(self, x):
        general_feature_map = self.backbone(x)
        instance_class_map = self.instance_class_module(general_feature_map)
        cla_logist = self.cla_branch(instance_class_map)
        instance_map, pred_bbox = self.bbox_pred_branch(instance_class_map, cla_logist)
        return cla_logist, pred_bbox, instance_map
