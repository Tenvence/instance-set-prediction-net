import os

import albumentations as alb
import albumentations.augmentations.bbox_utils as bbox_utils
import torch
import torchvision.datasets as cv_datasets
from albumentations.pytorch.transforms import ToTensorV2

from datasets import load_img_target


class AugVocTrainDataset(cv_datasets.CocoDetection):
    def __init__(self, root, input_size_w=448, input_size_h=448, num_instances=50):
        super(AugVocTrainDataset, self).__init__(root=os.path.join(root, 'train'), annFile=os.path.join(root, 'train.json'))
        self.transform = alb.Compose([
            alb.Resize(width=input_size_w, height=input_size_h),
            # alb.RandomSizedBBoxSafeCrop(width=input_size_w, height=input_size_h),
            # alb.HorizontalFlip(p=0.5),
            # alb.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0., p=0.8),
            alb.Normalize(),
            ToTensorV2()
        ], bbox_params=alb.BboxParams(format='coco', label_fields=['class_labels']))
        self.num_instances = num_instances
        self.h = input_size_h
        self.w = input_size_w

    def __getitem__(self, index):
        img, target = load_img_target(self, index)

        bboxes = [obj['bbox'] for obj in target]
        category_ids = [obj['category_id'] for obj in target]

        transformed = self.transform(image=img, bboxes=bboxes, class_labels=category_ids)
        img = transformed['image']

        category_ids = transformed['class_labels']
        category_ids = self._pad_category_ids(torch.as_tensor(category_ids))

        bboxes = transformed['bboxes']
        bbox_masks = torch.zeros((len(bboxes), int(self.h / 8), int(self.w / 8)))
        for i in range(len(bboxes)):
            x_min, y_min, bw, bh = bboxes[i]
            x_min, y_min, bw, bh = int(x_min / 8), int(y_min / 8), int(bw / 8), int(bh / 8)
            bbox_masks[i, y_min:y_min + bh, x_min:x_min + bw] = 1.
        bbox_masks = self._pad_bbox_masks(bbox_masks)

        return img, category_ids, bbox_masks

    @staticmethod
    def _norm_bboxes(bboxes, img):
        _, h, w = img.shape
        bboxes = bbox_utils.convert_bboxes_to_albumentations(bboxes, source_format='coco', rows=h, cols=w)
        bboxes = bbox_utils.convert_bboxes_from_albumentations(bboxes, target_format='yolo', rows=h, cols=w)
        return bboxes

    def _pad_category_ids(self, category_ids):
        pad_category_ids = torch.zeros(self.num_instances)
        pad_category_ids[:category_ids.shape[0]] = category_ids
        return pad_category_ids

    def _pad_bbox_masks(self, bbox_masks):
        pad_bbox_masks = torch.ones((self.num_instances, int(self.h / 8), int(self.w / 8)))
        pad_bbox_masks[:bbox_masks.shape[0], ...] = bbox_masks
        pad_bbox_masks[bbox_masks.shape[0]:, ...] = 1. - torch.sum(bbox_masks, dim=0).bool().float()
        return pad_bbox_masks
