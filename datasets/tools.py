import torchvision.datasets as cv_datasets
import cv2
import os

CLASSES = [
    '__no_instance__',
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def load_img_target(dataset: cv_datasets.CocoDetection, index: int):
    coco = dataset.coco
    img_id = dataset.ids[index]
    ann_ids = coco.getAnnIds(imgIds=img_id)

    target = coco.loadAnns(ann_ids)
    img = cv2.imread(os.path.join(dataset.root, coco.loadImgs(img_id)[0]['file_name']))

    return img, target
