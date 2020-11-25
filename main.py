import random
import json
import os

import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torch.utils.data
import torch.utils.tensorboard as tensorboard
import tqdm
import pycocotools.cocoeval as coco_eval

import utils.lr_lambda as lr_lambda
from datasets import AugVocTrainDataset, AugVocValDataset, CLASSES
from model import InstanceSetPredictionNet, SetCriterion, HungarianMatcher

rand_seed = 423
random.seed(rand_seed)
np.random.seed(rand_seed)
torch.random.manual_seed(rand_seed)


def recover_bboxes(bboxes, ow, oh):
    bboxes[:, [0, 2]] *= ow
    bboxes[:, [1, 3]] *= oh
    bboxes[:, :2] -= bboxes[:, 2:] / 2
    return bboxes


def train(model, optimizer, criterion, lr_scheduler, data_loader, logger, epoch_idx):
    print('Training Epoch: %d' % (epoch_idx + 1))
    model.train()
    processor = tqdm.tqdm(data_loader)
    losses = []
    label_losses = []
    bbox_losses = []
    for img, pad_category_ids, pad_bboxes in processor:
        img, pad_category_ids, pad_bboxes = img.cuda(), pad_category_ids.cuda(), pad_bboxes.cuda()

        cla_logist, pred_bbox = model(img)
        loss, label_loss, bbox_loss = criterion(cla_logist, pred_bbox, pad_category_ids, pad_bboxes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss))
        label_losses.append(float(label_loss))
        bbox_losses.append(float(bbox_loss))
        processor.set_description('cur_loss: %.4f, avg_loss: %.4f' % (float(loss), sum(losses) / len(losses)))
    logger.add_scalar('loss', sum(losses) / len(losses), epoch_idx)
    logger.add_scalar('loss/label_loss', sum(label_losses) / len(label_losses), epoch_idx)
    logger.add_scalar('loss/bbox_loss', sum(bbox_losses) / len(bbox_losses), epoch_idx)
    lr_scheduler.step()
    print()


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

        with torch.no_grad():
            # cla_logist: [B, num_instances, num_classes]; pred_bbox: [B, num_instance, 4]
            cla_logist, pred_bbox = model(img)
            cla_logist = func.softmax(cla_logist, dim=-1)

            cla_conf, cla_pred = torch.max(cla_logist, dim=-1)  # [B, num_instances]

        cla_conf, cla_pred, pred_bbox = cla_conf.cpu().numpy(), cla_pred.cpu().numpy(), pred_bbox.cpu().numpy()
        batch_size, num_instances = cla_conf.shape

        for batch_idx in range(batch_size):
            recovered_bboxes = recover_bboxes(pred_bbox[batch_idx], oh=org_h[batch_idx], ow=org_w[batch_idx])
            for instance_idx in range(num_instances):
                if cla_pred[batch_idx, instance_idx] != 0:
                    pred_instances.append({
                        'image_id': int(img_id[batch_idx]),
                        'category_id': int(cla_pred[batch_idx, instance_idx]),
                        'bbox': [float(str('%.1f' % coord)) for coord in recovered_bboxes[instance_idx].tolist()],
                        'score': float(str('%.1f' % cla_conf[batch_idx, instance_idx]))
                    })

    val_res_file = os.path.join(res_file_path, 'val_%d.json' % (epoch_idx + 1))
    with open(val_res_file, 'w') as f:
        json.dump(pred_instances, f)

    evaluate(logger, coco_gt, val_res_file, epoch_idx)
    print()


def evaluate(logger, coco_gt, res_file, epoch_idx):
    coco_evaluator = coco_eval.COCOeval(cocoGt=coco_gt, cocoDt=coco_gt.loadRes(res_file), iouType='bbox')
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    ap, ap50, ap75, ap_s, ap_m, ap_l, ar1, ar10, ar100, ar_s, ar_m, ar_l = coco_evaluator.stats
    logger.add_scalar('val/ap', ap, epoch_idx)
    logger.add_scalar('val/ap50', ap50, epoch_idx)
    logger.add_scalar('val/ap75', ap75, epoch_idx)
    print('AP: %.1f, AP50: %.1f, AP75: %.1f' % (ap * 100, ap50 * 100, ap75 * 100))


def __main__():
    name = 'larger_batch_size_warm_up_cosine_lr_0.05'
    dataset_root = '../../DataSet/AugVoc2012'
    output_path = os.path.join('./output', name)
    input_size_w, input_size_h = 448, 448
    num_instances = 50
    warm_up_epochs = 5
    epochs = 200
    logger = tensorboard.SummaryWriter(comment=name)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    train_dataset = AugVocTrainDataset(dataset_root, input_size_w, input_size_h, num_instances)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, num_workers=4, shuffle=True, drop_last=True)

    val_dataset = AugVocValDataset(dataset_root, input_size_w, input_size_h)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, num_workers=4, shuffle=False)

    instance_set_prediction_net = nn.DataParallel(InstanceSetPredictionNet(num_classes=len(CLASSES), num_instances=num_instances)).cuda().train()
    optimizer = optim.SGD(instance_set_prediction_net.module.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda.get_warm_up_cosine_lr_lambda(warm_up_epochs, cosine_epochs=epochs - warm_up_epochs))
    matcher = HungarianMatcher(class_weight=1., giou_weight=2.)
    criterion = SetCriterion(matcher, no_instance_coef=0.2, label_loss_coef=1., giou_loss_coef=2.)

    for epoch_idx in range(warm_up_epochs + epochs):
        train(instance_set_prediction_net, optimizer, criterion, lr_scheduler, train_dataloader, logger, epoch_idx)
        val(instance_set_prediction_net, val_dataloader, val_dataset.coco, logger, output_path, epoch_idx)


if __name__ == '__main__':
    __main__()
