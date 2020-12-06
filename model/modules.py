import torch
import torch.nn as nn
import torchvision.ops as cv_ops


class StageBackbone(nn.Module):
    def __init__(self, backbone):
        super(StageBackbone, self).__init__()
        backbone_stage_list = list(backbone.children())[:-2]
        self.backbone_c5 = backbone_stage_list[-1]
        self.backbone_c4 = backbone_stage_list[-2]
        self.backbone_c3 = backbone_stage_list[-3]
        self.backbone_rest = nn.Sequential(*list(backbone_stage_list[:-3]))

    def forward(self, x):
        x = self.backbone_rest(x)
        c3 = self.backbone_c3(x)
        c4 = self.backbone_c4(c3)
        c5 = self.backbone_c5(c4)
        return c3, c4, c5


class PathAggregationNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(PathAggregationNetwork, self).__init__()

        self.fpn = cv_ops.FeaturePyramidNetwork(in_channels_list, out_channels)
        self.pan = cv_ops.FeaturePyramidNetwork([out_channels, out_channels, out_channels], out_channels)

    def forward(self, c3, c4, c5):
        fpn_out = self.fpn({'p3': c3, 'p4': c4, 'p5': c5})
        pan_out = self.pan({'q5': fpn_out['p5'], 'q4': fpn_out['p4'], 'q3': fpn_out['p3']})
        q3, q4, q5 = pan_out['q3'], pan_out['q4'], pan_out['q5']
        return q3, q4, q5


class InstanceClassModule(nn.Module):
    def __init__(self, num_instances, num_classes, in_channels):
        super(InstanceClassModule, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=(num_instances * num_classes), kernel_size=1, bias=False)
        self.unfold = nn.Unflatten(dim=1, unflattened_size=(num_instances, num_classes))

    def forward(self, general_feature_map):
        general_feature_map = self.conv(general_feature_map)
        instance_class_map = self.unfold(general_feature_map)
        return instance_class_map


class ClaBranch(nn.Module):
    def __init__(self):
        super(ClaBranch, self).__init__()

    @staticmethod
    def forward(instance_class_map):
        instance_class_vec = torch.mean(instance_class_map, dim=[-1, -2])
        return instance_class_vec


class BboxPredBranch(nn.Module):
    def __init__(self, inner_dim):
        super(BboxPredBranch, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=inner_dim * inner_dim, out_features=inner_dim * inner_dim), nn.ReLU(),
            nn.Linear(in_features=inner_dim * inner_dim, out_features=inner_dim * inner_dim), nn.ReLU(),
            nn.Linear(in_features=inner_dim * inner_dim, out_features=inner_dim * inner_dim), nn.ReLU(),
            nn.Linear(in_features=inner_dim * inner_dim, out_features=inner_dim * inner_dim), nn.ReLU(),
            nn.Linear(in_features=inner_dim * inner_dim, out_features=4), nn.Sigmoid()
        )

    def forward(self, instance_class_map, cla_logist):
        _, _, _, h, w = instance_class_map.shape
        select_idx = torch.argmax(cla_logist, dim=-1, keepdim=True)[..., None, None].repeat(1, 1, 1, h, w)  # [B, I, 1] -> [B, I, 1, H, W]
        instance_map = torch.gather(instance_class_map, dim=2, index=select_idx).squeeze()  # [B, I, C, H, W] -> [B, I, 1, H, W] -> [B, I, H, W]
        flattened_instance_map = torch.flatten(instance_map, start_dim=2)  # [B, I, H, W] -> [B, I, HW]
        pred_bbox = self.mlp(flattened_instance_map)
        return instance_map, pred_bbox