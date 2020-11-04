import os

import albumentations as alb
import albumentations.augmentations.bbox_utils as bbox_utils
import cv2
import torch
import torchvision.datasets as cv_datasets


class AugVocTrainDataset(cv_datasets.CocoDetection):
    def __init__(self, root, trans_list, num_instances):
        super(AugVocTrainDataset, self).__init__(root=os.path.join(root, 'train'), annFile=os.path.join(root, 'train.json'))
        self.transform = alb.Compose(trans_list, bbox_params=alb.BboxParams(format='coco', label_fields=['class_labels']))
        self.num_instances = num_instances

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)

        target = coco.loadAnns(ann_ids)
        img = cv2.imread(os.path.join(self.root, coco.loadImgs(img_id)[0]['file_name']))

        bboxes, classes = [], []
        for obj in target:
            classes.append(obj['category_id'])
            bboxes.append(obj['bbox'])

        transformed = self.transform(image=img, bboxes=bboxes, class_labels=classes)
        img = transformed['image']
        bboxes = transformed['bboxes']
        classes = transformed['class_labels']

        _, h, w = img.shape
        # [x_min, y_min, w, h] -> [normalized_center_x, normalized_center_y, normalized_w, normalized_h]
        bboxes = bbox_utils.convert_bboxes_to_albumentations(bboxes, source_format='coco', rows=h, cols=w)
        bboxes = bbox_utils.convert_bboxes_from_albumentations(bboxes, target_format='yolo', rows=h, cols=w)

        bboxes = torch.as_tensor(bboxes)
        classes = torch.as_tensor(classes)

        pad_bboxes = torch.zeros((self.num_instances, 4))
        pad_classes = torch.zeros(self.num_instances)

        pad_bboxes[:len(target), :] = bboxes
        pad_classes[:len(target)] = classes

        return img, pad_classes, pad_classes
