import torch
import torchvision.datasets as cv_datasets

from .voc_tools import CLASSES


class VocMultiCla(cv_datasets.VOCDetection):
    def __init__(self, root):
        super(VocMultiCla, self).__init__(root)

    def __getitem__(self, index):
        img, anno = super(VocMultiCla, self).__getitem__(index)
        cla_labels = []
        for obj in anno['annotation']['object']:
            cla_labels.append(CLASSES.index(obj['name']))
        return img, torch.as_tensor(cla_labels)
