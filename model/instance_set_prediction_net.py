import torch.nn as nn
import torchvision.models as cv_models

from .modules import StageBackbone, PathAggregationNetwork, InstanceClassModule, ClaBranch, BboxPredBranch


class InstanceSetPredictionNet(nn.Module):
    def __init__(self, num_classes, num_instances):
        super(InstanceSetPredictionNet, self).__init__()

        self.backbone = StageBackbone(cv_models.resnet50(pretrained=True))
        self.pan = PathAggregationNetwork(in_channels_list=[512, 1024, 2048], out_channels=2048)
        self.instance_class_module = InstanceClassModule(num_instances, num_classes, in_channels=2048)
        self.cla_branch = ClaBranch()
        self.bbox_pred_branch = BboxPredBranch(inner_dim=14)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        q3, q4, q5 = self.pan(c3, c4, c5)

        instance_class_map = self.instance_class_module(q5)
        cla_logist = self.cla_branch(instance_class_map)
        instance_map, pred_bbox = self.bbox_pred_branch(instance_class_map, cla_logist)

        return cla_logist, pred_bbox, instance_map
