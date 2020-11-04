import albumentations as alb
import torch.utils.data
import tqdm
from albumentations.pytorch.transforms import ToTensorV2

from datasets.aug_voc_train_dataset import AugVocTrainDataset

if __name__ == '__main__':
    num_instances = 100
    dataset_root = '../../DataSet/AugVoc2012'
    train_trans_list = [
        alb.RandomSizedBBoxSafeCrop(width=448, height=448),
        alb.HorizontalFlip(p=0.5),
        alb.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0., p=0.8),
        alb.Normalize(),
        ToTensorV2()
    ]
    train_dataset = AugVocTrainDataset(root=dataset_root, trans_list=train_trans_list, num_instances=num_instances)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=16, shuffle=True, drop_last=True)

    processor = tqdm.tqdm(train_dataloader)
    for idx, classes, bboxes in processor:
        pass
