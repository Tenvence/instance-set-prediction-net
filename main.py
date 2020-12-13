import json
import os
import random

import numpy as np
import pycocotools.cocoeval as coco_eval
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.utils.tensorboard as tensorboard
import tqdm

import utils.bbox_ops as bbox_ops
import utils.lr_lambda as lr_lambda
from datasets import AugVocTrainDataset, AugVocValDataset, CLASSES
from model import InstanceSetPredictionNet, SetCriterion, HungarianMatcher


def update_logger(logger, super_tag, tag_scaler_dict, idx):
    for tag, scaler in tag_scaler_dict.items():
        logger.add_scalar(super_tag + '/' + tag, scaler, idx)


def train(model, optimizer, criterion, lr_scheduler, data_loader, logger, res_file_path, epoch_idx):
    print('Training Epoch: %d' % (epoch_idx + 1))
    model.train()
    processor = tqdm.tqdm(data_loader)
    losses = []
    label_losses = []
    bbox_losses = []
    scaler = amp.GradScaler()
    for img, pad_category_ids, pad_bboxes in processor:
        img, pad_category_ids, pad_bboxes = img.cuda(), pad_category_ids.cuda(), pad_bboxes.cuda()

        cla_logist, pred_bbox, _ = model(img)
        label_loss, bbox_loss = criterion(cla_logist, pred_bbox, pad_category_ids, pad_bboxes)
        loss = criterion.class_weight * label_loss + criterion.giou_weight * bbox_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss))
        label_losses.append(float(label_loss))
        bbox_losses.append(float(bbox_loss))
        processor.set_description('cur_loss: %.4f, avg_loss: %.4f' % (float(loss), sum(losses) / len(losses)))
    torch.save(model.module, os.path.join(res_file_path, 'model.pkl'))
    torch.save(model.module.state_dict(), os.path.join(res_file_path, 'param.pth'))
    update_logger(logger, super_tag='loss', tag_scaler_dict={
        'total-loss': sum(losses) / len(losses),
        'label-loss': sum(label_losses) / len(label_losses),
        'bbox-loss': sum(bbox_losses) / len(bbox_losses)
    }, idx=epoch_idx)
    lr_scheduler.step()
    print()


@torch.no_grad()
def val(model, data_loader, coco_gt, logger, res_file_path, epoch_idx):
    print('Val Epoch: %d' % (epoch_idx + 1))
    model.eval()
    processor = tqdm.tqdm(data_loader)
    pred_instances = []
    for img, img_id, org_h, org_w in processor:
        img = img.cuda()
        img_id = img_id.numpy()
        org_h = org_h.numpy()
        org_w = org_w.numpy()

        cla_logist, pred_bbox, instance_map = model(img)  # cla_logist: [B, I, C]; pred_bbox: [B, I, 4]; instance_map: [B, I, H, W]
        cla_conf, cla_pred = torch.max(cla_logist.softmax(dim=-1), dim=-1)  # [B, I]
        instance_map = instance_map > 0.3

        cla_conf, cla_pred, pred_bbox = cla_conf.cpu().numpy(), cla_pred.cpu().numpy(), pred_bbox.cpu().numpy()
        batch_size, num_instances = cla_conf.shape

        for batch_idx in range(batch_size):
            recovered_bboxes = bbox_ops.recover_bboxes(pred_bbox[batch_idx], oh=org_h[batch_idx], ow=org_w[batch_idx])
            for instance_idx in range(num_instances):
                if cla_pred[batch_idx, instance_idx] != 0:
                    pred_instances.append({
                        'image_id': int(img_id[batch_idx]),
                        'category_id': int(cla_pred[batch_idx, instance_idx]),
                        'bbox': [float(str('%.1f' % coord)) for coord in recovered_bboxes[instance_idx].tolist()],
                        'score': float(str('%.1f' % cla_conf[batch_idx, instance_idx]))
                    })

    val_res_file = os.path.join(res_file_path, 'val.json')
    with open(val_res_file, 'w') as f:
        json.dump(pred_instances, f)

    evaluate(logger, coco_gt, val_res_file, epoch_idx)
    print()


def evaluate(logger, coco_gt, res_file, epoch_idx):
    coco_evaluator = coco_eval.COCOeval(cocoGt=coco_gt, cocoDt=coco_gt.loadRes(res_file), iouType='bbox')
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    ap, ap50, ap75, ap_s, ap_m, ap_l = coco_evaluator.stats[:6]
    update_logger(logger, super_tag='bbox-val', tag_scaler_dict={
        'AP': ap * 100, 'AP@50': ap50 * 100, 'AP@75': ap75 * 100,
        'AP-S': ap_s * 100, 'AP-M': ap_m * 100, 'AP-L': ap_l * 100
    }, idx=epoch_idx)


def __main__():
    name = 'sigmoid_in_icm'
    dataset_root = '../../DataSet/AugVoc2012'
    output_path = os.path.join('./output', name)
    input_size_w, input_size_h = 448, 448
    num_instances = 50
    warm_up_epochs = 5
    epochs = 200
    batch_size = 256
    num_workers = 4
    logger = tensorboard.SummaryWriter(comment=name)

    rand_seed = 423
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.random.manual_seed(rand_seed)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    train_dataset = AugVocTrainDataset(dataset_root, input_size_w, input_size_h, num_instances)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)

    val_dataset = AugVocValDataset(dataset_root, input_size_w, input_size_h)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    instance_set_prediction_net = nn.DataParallel(InstanceSetPredictionNet(num_classes=len(CLASSES), num_instances=num_instances)).cuda().train()
    optimizer = optim.SGD(instance_set_prediction_net.module.parameters(), lr=0.02, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda.get_warm_up_cosine_lr_lambda(warm_up_epochs, cosine_epochs=epochs))
    matcher = HungarianMatcher(class_weight=1., giou_weight=2.)
    criterion = SetCriterion(matcher, no_instance_coef=0.2)

    for epoch_idx in range(warm_up_epochs + epochs):
        train(instance_set_prediction_net, optimizer, criterion, lr_scheduler, train_dataloader, logger, output_path, epoch_idx)
        val(instance_set_prediction_net, val_dataloader, val_dataset.coco, logger, output_path, epoch_idx)


if __name__ == '__main__':
    __main__()
