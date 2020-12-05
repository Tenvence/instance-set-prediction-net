# Instance Set Prediction Net (ISPNet)

## Dataset

- Augmented PASCAL VOC 2012 
    - Extent with [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html) training set
    - We provide 2 download links:
        - [Baidu Cloud Disk](https://pan.baidu.com/s/1WyqdEYSVGORhDlOLYk83cA) with the extraction code `i8ia`;
        - [Google Drive](https://drive.google.com/file/d/16rE2PtDjG1XrePESj_z66mU5ZEMW9oCu/view?usp=sharing).

## Models

| Model       | Publish | Sup. | Backbone | mAP25 | mAP50 | mAP70 | mAP75 | ABO  |
|:-----------:|:-------:|:----:|:--------:|:-----:|:-----:|:-----:|:-----:|:----:|
| Mask R-CNN  | ICCV'17 | F    | r101     | 76.7  | 67.9  | 52.5  | 44.9  | -    |
| PRM         | CVPR'18 | I    | r50      | 44.3  | 26.8  | -     | 9.0   | 37.6 |
| IAM         | CVPR'19 | I    | r50      | 45.9  | 28.8  | -     | 11.9  | 41.9 |
| OCIS        | CVPR'19 | I    | r50      | 48.5  | 30.2  | -     | 14.4  | 44.3 |
| Label-PEnet | ICCV'19 | I    | r50      | 49.1  | 30.2  | -     | 12.9  | 41.4 |
| WISE        | BMCV'19 | I    | r50      | 49.2  | 41.7  | -     | 23.7  | 55.2 |
| IRNet       | CVPR'19 | I    | r50      | -     | 46.7  | -     | 23.5  | -    |
| LACI        | ECCV'20 | I    | r50      | 59.7  | 50.9  | 30.2  | 28.5  | -    |
| SDI         | CVPR'17 | B    | r101     | -     | 44.8  | -     | 16.3  | 49.1 |
| BBTP        | NIPS'19 | B    | r101     | 75.0  | 58.9  | 30.4  | 21.6  | -    |
| LACI        | ECCV'20 | B    | r101     | 73.8  | 58.2  | 34.3  | 32.1  | -    |

## Fully-Supervised Instance Segmentation Method (Mask R-CNN in [MMDetection](https://github.com/open-mmlab/mmdetection))

1. install MMDetection
2. modify 
- `/mmdet/datasets/coco.py`
    - **Line 32-45**: modify `CLASSES=('person', ..., 'toothbrush')` to `CLASSES=('aeroplane', ..., 'tvmonitor')` (refer to the 10th-14th lines in `/mmdet/datasets/voc.py`)
- `/configs/_base_/datasets/coco_instance.py`
    - **Line 2**: modify `data_root` from `data/coco` to your dataset path （absolute path is recommended)
    - **Line 8 & Line 19**: modify `img_scale=(1333, 800)` to `img_scale=(448, 448)`
    - **Line 35 & Line 36**:
    - **Line 40 & Line 45**: modify `ann_file=data_root + 'annotations/instances_val2017.json'` to `ann_file=data_root + 'val.json'`
    - **Line 41 & Line 46**: modify `img_prefix=data_root + 'val2017/'` to `img_prefix=data_root + 'val/'`
- `/config/_base_/models/mask_rcnn_r50_fpn.py`
    - **Line 47 & Line 66**: modify `num_classes=80` to `num_classes=20`

