import torch
import torch.nn as nn
import torchvision.models as cv_models

from model.transformer import Transformer
from model.frozen_batch_norm import FrozenBatchNorm2d
from model.sine_position_embedding import SinePositionEmbedding


class InstanceTransformer(nn.Module):
    def __init__(self, num_instances, num_classes, d_model):
        super(InstanceTransformer, self).__init__()

        backbone_channels = 2048
        self.backbone = nn.Sequential(*list(cv_models.resnet50(pretrained=True, norm_layer=FrozenBatchNorm2d).children())[:-2])
        self.conv = nn.Conv2d(in_channels=backbone_channels, out_channels=d_model, kernel_size=1)

        self.transformer = Transformer(d_model=d_model)
        self.instance_queries = nn.Embedding(num_embeddings=num_instances, embedding_dim=d_model)
        self.pos_embedding = SinePositionEmbedding(d_model)

        self.class_pred_head = nn.Linear(in_features=d_model, out_features=num_classes)
        self.bbox_pred_head = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_model), nn.ReLU(),
            nn.Linear(in_features=d_model, out_features=d_model), nn.ReLU(),
            nn.Linear(in_features=d_model, out_features=4), nn.Sigmoid()
        )

    def forward(self, x):
        cnn_features = self.backbone(x)
        cnn_features = self.conv(cnn_features)
        b, _, h, w = cnn_features.shape

        src = torch.flatten(cnn_features, start_dim=2).permute(2, 0, 1)
        pos_embed = self.pos_embedding(torch.zeros((b, h, w)).cuda().bool()).flatten(start_dim=2).permute(2, 0, 1)
        query_embed = self.instance_queries.weight.unsqueeze(dim=1).repeat(1, b, 1)
        tr_out = self.transformer(src, pos_embed, query_embed)

        head_inp = tr_out.permute(1, 0, 2)
        return self.class_pred_head(head_inp), self.bbox_pred_head(head_inp)
