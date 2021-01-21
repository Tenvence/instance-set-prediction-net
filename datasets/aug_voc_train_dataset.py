import os

import albumentations as alb
import torch
import torchvision.datasets as cv_datasets
from albumentations.pytorch.transforms import ToTensorV2

from datasets import load_img_target
import utils.bbox_ops as bbox_ops


class AugVocTrainDataset(cv_datasets.CocoDetection):
    def __init__(self, root, input_size_w, input_size_h, num_instances):
        super(AugVocTrainDataset, self).__init__(root=os.path.join(root, 'train'), annFile=os.path.join(root, 'train.json'))
        self.transform = alb.Compose([
            alb.RandomSizedBBoxSafeCrop(width=input_size_w, height=input_size_h),
            alb.HorizontalFlip(p=0.5),
            alb.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0., p=0.8),
            alb.Normalize(),
            ToTensorV2()
        ], bbox_params=alb.BboxParams(format='coco', label_fields=['class_labels']))
        self.num_instances = num_instances
        self.h = input_size_h
        self.w = input_size_w

    def __getitem__(self, index):
        img, target = load_img_target(self, index)
        h, w, _ = img.shape

        bboxes = [obj['bbox'] for obj in target]
        category_ids = [obj['category_id'] for obj in target]

        transformed = self.transform(image=img, bboxes=bboxes, class_labels=category_ids)
        img = transformed['image']
        category_ids = self._pad_category_ids(torch.as_tensor(transformed['class_labels']))
        bboxes = self._pad_bboxes(bbox_ops.normalize_bboxes(transformed['bboxes'], h, w))

        return img, category_ids, bboxes

    def _pad_category_ids(self, category_ids):
        pad_category_ids = torch.zeros(self.num_instances)
        pad_category_ids[:category_ids.shape[0]] = category_ids
        return pad_category_ids

    def _pad_bboxes(self, bboxes):
        pad_bboxes = torch.zeros((self.num_instances, 4))
        pad_bboxes[:bboxes.shape[0], :] = bboxes
        return pad_bboxes
