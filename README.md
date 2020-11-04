# Instance Set Prediction Net (ISPNet)

## Dataset

- Augmented PASCAL VOC 2012 
    - Extent with SBD training set
    - We provide 2 download links:
        - [Baidu Cloud Disk](https://pan.baidu.com/s/1WyqdEYSVGORhDlOLYk83cA) with extraction code **i8ia**;
        - [Google Drive]()

## Models

| Model       | Pubish  | Supervision | Backbone | mAP25 | mAP50 | mAP70 | mAP75 | ABO  | Link |
|:-----------:|:-------:|:-----------:|:--------:|:-----:|:-----:|:-----:|:-----:|:----:|:----:|
| Mask R-CNN  | ICCV'17 | F           | r101     | 76.7  | 67.9  | 52.5  | 44.9  | -    |      |
| PRM         | CVPR'18 | I           | r50      | 44.3  | 26.8  | -     | 9.0   | 37.6 ||
| IAM         | CVPR'19 | I           | r50      | 45.9  | 28.8  | -     | 11.9  | 41.9 ||
| OCIS        | CVPR'19 | I           | r50      | 48.5  | 30.2  | -     | 14.4  | 44.3 ||
| Label-PEnet | ICCV'19 | I           | r50      | 49.1  | 30.2  | -     | 12.9  | 41.4 ||
| WISE        | BMCV'19 | I           | r50      | 49.2  | 41.7  | -     | 23.7  | 55.2 ||
| IRNet       | CVPR'19 | I           | r50      | -     | 46.7  | -     | 23.5  | -    ||
| LACI        | ECCV'20 | I           | r50      | 59.7  | 50.9  | 30.2  | 28.5  | -    ||
| SDI         | CVPR'17 | B           | r101     | -     | 44.8  | -     | 16.3  | 49.1 ||
| BBTP        | NIPS'19 | B           | r101     | 75.0  | 58.9  | 30.4  | 21.6  | -    ||
| LACI        | ECCV'20 | B           | r101     | 73.8  | 58.2  | 34.3  | 32.1  | -    ||