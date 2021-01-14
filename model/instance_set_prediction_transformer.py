import torch
import torch.nn as nn


class InstanceSetPredictionTransformer(nn.Module):
    def __init__(self, num_instances, num_classes, patch_num, patch_size, d_model):
        super(InstanceSetPredictionTransformer, self).__init__()

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.ffn = nn.Linear(in_features=3 * patch_size * patch_size, out_features=d_model)

        self.transformer = nn.Transformer(d_model)
        self.instance_queries = nn.Embedding(num_embeddings=num_instances, embedding_dim=d_model)
        self.pos_embedding = nn.Parameter(torch.rand(patch_num, 1, d_model))

        self.class_pred_head = nn.Linear(in_features=d_model, out_features=num_classes)
        self.bbox_pred_head = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_model), nn.ReLU(),
            nn.Linear(in_features=d_model, out_features=d_model), nn.ReLU(),
            nn.Linear(in_features=d_model, out_features=d_model), nn.ReLU(),
            nn.Linear(in_features=d_model, out_features=d_model), nn.ReLU(),
            nn.Linear(in_features=d_model, out_features=4), nn.Sigmoid()
        )

    def forward(self, x):
        patches = self.unfold(x).permute(0, 2, 1)
        ffn_patches = self.ffn(patches)
        tr_inp = ffn_patches.permute(1, 0, 2)

        tr_out = self.transformer(src=tr_inp + self.pos_embedding, tgt=self.instance_queries.weight.unsqueeze(dim=1).repeat(1, x.shape[0], 1))
        head_inp = tr_out.permute(1, 0, 2)

        cla_logist = self.class_pred_head(head_inp)
        bbox_pred = self.bbox_pred_head(head_inp)

        return cla_logist, bbox_pred
