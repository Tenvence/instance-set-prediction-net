import torch.nn.functional as func
import torchvision.datasets as cv_datasets
from PIL import Image

from .voc_tools import generate_mask


class VocInstanceSeg(cv_datasets.VOCSegmentation):
    def __init__(self, root):
        super(VocInstanceSeg, self).__init__(root)

    def __getitem__(self, index):
        img, seg_mask = super(VocInstanceSeg, self).__getitem__(index)
        instance_mask = Image.open(self.masks[index].replace('SegmentationClass', 'SegmentationObject'))

        seg_mask = generate_mask(seg_mask)
        instance_mask = func.one_hot(generate_mask(instance_mask)).permute(2, 0, 1)[1:, ...]  # [H, W] -> [H, W, N] -> [N, H, W] & remove background
        instance_mask *= seg_mask  # keep classification information

        return img, seg_mask, instance_mask
