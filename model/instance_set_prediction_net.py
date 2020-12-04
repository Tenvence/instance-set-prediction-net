import torch
import torch.nn as nn
import torchvision.models as cv_models


class InstanceSetPredictionNet(nn.Module):
    def __init__(self, num_classes, num_instances):
        super(InstanceSetPredictionNet, self).__init__()

        self.num_classes = num_classes
        self.num_instances = num_instances

        self.backbone = nn.Sequential(*list(cv_models.resnet50(pretrained=True).children())[:-2])
        self.instance_class_conv = nn.Conv2d(in_channels=2048, out_channels=(num_instances * num_classes), kernel_size=1, bias=False)
        self.instance_class_dim_separator = nn.Unflatten(dim=1, unflattened_size=(num_instances, num_classes))

        self.cla_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            self.instance_class_dim_separator
        )

        inner_dim = 14
        self.bbox_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(inner_dim, inner_dim)),
            nn.Flatten(start_dim=2),
            nn.Linear(in_features=inner_dim * inner_dim, out_features=inner_dim * inner_dim), nn.ReLU(),
            nn.Linear(in_features=inner_dim * inner_dim, out_features=inner_dim * inner_dim), nn.ReLU(),
            nn.Linear(in_features=inner_dim * inner_dim, out_features=4), nn.Sigmoid()
        )

    def forward(self, x):
        general_feature_map = self.backbone(x)
        instance_class_map = self.instance_class_conv(general_feature_map)
        cla_logist = self.cla_branch(instance_class_map)
        _, cla_idx = torch.max(cla_logist, dim=-1, keepdim=True)  # index of maximum logist, [B, I, 1]

        instance_class_map = self.instance_class_dim_separator(instance_class_map)  # [B, IC, H, W] -> [B, I, C, H, W]
        _, _, _, h, w = instance_class_map.shape
        repeated_cla_idx = cla_idx[..., None, None].repeat(1, 1, 1, h, w)  # [B, I, 1] -> [B, I, 1, H, W]
        instance_map = torch.gather(instance_class_map, dim=2, index=repeated_cla_idx).squeeze()  # [B, I, C, H, W] -> [B, I, 1, H, W] -> [B, I, H, W]

        pred_bbox = self.bbox_branch(instance_map)

        return cla_logist, pred_bbox, instance_map
