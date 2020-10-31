import torch
import torchvision.transforms.functional as cv_func

CLASSES = (
    '__background__',
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
)


def generate_mask(mask):
    mask = cv_func.to_tensor(mask)
    mask = torch.where(torch.eq(mask, 1.), torch.zeros_like(mask), mask) * 255.  # remove boundary & recover index
    return mask.squeeze().long()  # [1, H, W] -> [H, W]
