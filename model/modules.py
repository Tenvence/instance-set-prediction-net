import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.ops as cv_ops


class StageBackbone(nn.Module):
    def __init__(self, backbone):
        super(StageBackbone, self).__init__()
        backbone_stage_list = list(backbone.children())[:-2]
        self.backbone_c5 = backbone_stage_list[-1]
        self.backbone_c4 = backbone_stage_list[-2]
        self.backbone_c3 = backbone_stage_list[-3]
        self.backbone_c2 = backbone_stage_list[-4]
        self.backbone_c1 = nn.Sequential(*list(backbone_stage_list[:-4]))

    def forward(self, x):
        c1 = self.backbone_c1(x)
        c2 = self.backbone_c2(c1)
        c3 = self.backbone_c3(c2)
        c4 = self.backbone_c4(c3)
        c5 = self.backbone_c5(c4)
        return c1, c2, c3, c4, c5


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.fpn = cv_ops.FeaturePyramidNetwork(in_channels_list, out_channels)

    def forward(self, feature_maps):
        fpn_out = self.fpn({'p%d' % idx: feature_map for idx, feature_map in enumerate(feature_maps)})
        return list(fpn_out.values())[0]


class InstanceSeparator(nn.Module):
    def __init__(self, num_instances, num_classes, d_model):
        super(InstanceSeparator, self).__init__()

        self.num_instances = num_instances
        self.num_classes = num_classes

        # self.ins_sep_conv_list = nn.ModuleList([nn.Sequential(
        #     nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1, bias=False),
        #     nn.SyncBatchNorm(num_features=d_model),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, bias=False),
        #     nn.SyncBatchNorm(num_features=d_model),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=d_model, out_channels=num_classes, kernel_size=1, bias=False),
        #     nn.SyncBatchNorm(num_features=num_classes),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, padding=1, bias=False)
        # ) for _ in range(num_instances)])
        self.ins_sep_conv_list = nn.ModuleList([nn.Conv2d(in_channels=d_model, out_channels=num_classes, kernel_size=1, bias=False) for _ in range(num_instances)])

    def forward(self, general_feature_map):
        class_map_list = []
        for m in self.ins_sep_conv_list:
            class_map = m(general_feature_map)
            class_map_list.append(class_map)
        instance_class_map = torch.stack(class_map_list, dim=1)

        return instance_class_map


class InstanceMapRefiner(nn.Module):
    def __init__(self, d_model, num_instances, num_classes):
        super(InstanceMapRefiner, self).__init__()

        self.num_instances = num_instances
        self.num_classes = num_classes

        self.img_conv = nn.Conv2d(in_channels=3, out_channels=d_model, kernel_size=1, bias=False)
        self.c1_conv = nn.Conv2d(in_channels=64, out_channels=d_model, kernel_size=1, bias=False)
        self.c2_conv = nn.Conv2d(in_channels=256, out_channels=d_model, kernel_size=1, bias=False)
        self.aux_conv = nn.Conv2d(in_channels=3 * d_model, out_channels=num_instances, kernel_size=1, bias=False)

        self.instance_attention_conv = nn.Conv2d(in_channels=num_instances, out_channels=num_instances, kernel_size=1, bias=False)
        self.spatial_attention_conv = nn.Conv2d(in_channels=num_instances, out_channels=num_instances, kernel_size=1, bias=False)

    def forward(self, instance_map, img_feature, c1_feature, c2_feature):
        img_feature = self.img_conv(img_feature)  # [B, 3, H, W] -> [B, d, H, W]
        c1_feature = self.c1_conv(c1_feature)  # [B, 64, H, W] -> [B, d, H, W]
        c2_feature = self.c2_conv(c2_feature)  # [B, 256, H, W] -> [B, d, H, W]

        aux_feature = torch.cat([img_feature, c1_feature, c2_feature], dim=1)  # [B, 3*d, H, W]
        aux_feature = self.aux_conv(aux_feature)  # [B, 3*d, H, W] -> [B, I, H, W]

        instance_atten = self.instance_attention_conv(aux_feature).mean(dim=[-2, -1], keepdim=True)
        spatial_atten = self.spatial_attention_conv(aux_feature).mean(dim=-3, keepdim=True)

        refined_instance_map = instance_map * instance_atten + instance_map * spatial_atten

        return refined_instance_map
