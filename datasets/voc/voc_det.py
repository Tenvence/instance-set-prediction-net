import torch
import torchvision.datasets as cv_datasets
from .voc_tools import CLASSES


class VocDet(cv_datasets.VOCDetection):
    def __init__(self, root):
        super(VocDet, self).__init__(root)

    def __getitem__(self, index):
        img, anno = super(VocDet, self).__getitem__(index)
        cla_labels, bboxes = [], []
        for obj in anno['annotation']['object']:
            cla_labels.append(CLASSES.index(obj['name']))
            bboxes.append([float(val) for val in obj['bndbox'].values()])

        return img, torch.as_tensor(cla_labels), torch.as_tensor(bboxes)
