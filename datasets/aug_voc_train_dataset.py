import os

import albumentations as alb
import albumentations.augmentations.bbox_utils as bbox_utils
import torch
import torch.nn.functional
import torchvision.datasets as cv_datasets
from albumentations.pytorch.transforms import ToTensorV2

from datasets import load_img_target


class AugVocTrainDataset(cv_datasets.CocoDetection):
    def __init__(self, root, input_size_w=448, input_size_h=448, num_instances=50):
        super(AugVocTrainDataset, self).__init__(root=os.path.join(root, 'train'), annFile=os.path.join(root, 'train.json'))
        self.transform = alb.Compose([
            alb.RandomSizedBBoxSafeCrop(width=input_size_w, height=input_size_h),
            alb.HorizontalFlip(p=0.5),
            alb.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0., p=0.8),
            alb.Normalize(),
            ToTensorV2()
        ], bbox_params=alb.BboxParams(format='coco', label_fields=['class_labels']))
        self.num_instances = num_instances

    def __getitem__(self, index):
        img, target = load_img_target(self, index)

        bboxes = [obj['bbox'] for obj in target]
        category_ids = [obj['category_id'] for obj in target]

        transformed = self.transform(image=img, bboxes=bboxes, class_labels=category_ids)
        img = transformed['image']
        bboxes = transformed['bboxes']
        category_ids = transformed['class_labels']

        bboxes = self._norm_bboxes(bboxes, img)  # [x_min, y_min, w, h] -> [normalized_center_x, normalized_center_y, normalized_w, normalized_h]

        category_ids = torch.as_tensor(category_ids)
        bboxes = torch.as_tensor(bboxes)
        pad_category_ids = self._pad_category_ids(category_ids)
        pad_bboxes = self._pad_bboxes(bboxes)

        return img, pad_category_ids, pad_bboxes

    @staticmethod
    def _norm_bboxes(bboxes, img):
        _, h, w = img.shape
        bboxes = bbox_utils.convert_bboxes_to_albumentations(bboxes, source_format='coco', rows=h, cols=w)
        bboxes = bbox_utils.convert_bboxes_from_albumentations(bboxes, target_format='yolo', rows=h, cols=w)
        return bboxes

    def _pad_bboxes(self, bboxes):
        pad_bboxes = torch.zeros((self.num_instances, 4))
        pad_bboxes[:bboxes.shape[0], :] = bboxes
        return pad_bboxes

    def _pad_category_ids(self, category_ids):
        pad_category_ids = torch.zeros(self.num_instances)
        pad_category_ids[:category_ids.shape[0]] = category_ids
        return pad_category_ids
