import torchvision.datasets as cv_datasets

from .voc_tools import generate_mask


class VocSemanticSeg(cv_datasets.VOCSegmentation):
    def __init__(self, root):
        super(VocSemanticSeg, self).__init__(root)

    def __getitem__(self, index):
        img, seg_mask = super(VocSemanticSeg, self).__getitem__(index)
        seg_mask = generate_mask(seg_mask)
        return img, seg_mask
