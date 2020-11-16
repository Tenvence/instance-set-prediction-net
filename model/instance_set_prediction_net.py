import torch.nn as nn
import torchvision.models as cv_models


class InstanceSetPredictionNet(nn.Module):
    def __init__(self, num_classes, num_instances):
        super(InstanceSetPredictionNet, self).__init__()

        self.backbone = nn.Sequential(*list(cv_models.resnet50(pretrained=True).children())[:-2])
        self.instance_class_conv = nn.Conv2d(in_channels=2048, out_channels=(num_instances * num_classes), kernel_size=1, bias=False)

        self.cla_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Unflatten(dim=-1, unflattened_size=(num_instances, num_classes))
        )

        inner_dim = 10
        self.bbox_branch = nn.Sequential(
            nn.Conv2d(in_channels=(num_instances * num_classes), out_channels=num_instances, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=num_instances),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(inner_dim, inner_dim)),
            nn.Flatten(start_dim=2),
            nn.Linear(in_features=inner_dim * inner_dim, out_features=inner_dim * inner_dim), nn.ReLU(),
            nn.Linear(in_features=inner_dim * inner_dim, out_features=inner_dim * inner_dim), nn.ReLU(),
            nn.Linear(in_features=inner_dim * inner_dim, out_features=4), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        instance_cam = self.instance_class_conv(x)
        cla_logist = self.cla_branch(instance_cam)
        pred_bbox = self.bbox_branch(instance_cam)

        return cla_logist, pred_bbox
