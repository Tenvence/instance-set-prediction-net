import os

import albumentations as alb
import torchvision.datasets as cv_datasets
from albumentations.pytorch.transforms import ToTensorV2

from datasets import load_img_target


class AugVocValDataset(cv_datasets.CocoDetection):
    def __init__(self, root, input_size_w=448, input_size_h=448):
        super(AugVocValDataset, self).__init__(root=os.path.join(root, 'val'), annFile=os.path.join(root, 'val.json'))
        self.transform = alb.Compose([
            alb.Resize(width=input_size_w, height=input_size_h),
            alb.Normalize(),
            ToTensorV2()
        ])

    def __getitem__(self, index):
        img, target = load_img_target(self, index)
        original_h, original_w, _ = img.shape

        img = self.transform(image=img)['image']
        img_id = self.ids[index]

        return img, img_id, original_h, original_w
