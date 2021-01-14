import numpy as np
import pycocotools.mask as coco_mask
import torch
import torch.cuda.amp as amp
import torch.nn.functional as func


def train_one_epoch(model, optimizer, criterion, lr_scheduler, data_loader, dist_logger, epoch_idx):
    losses, cla_losses, mask_losses, aff_cla_losses, aff_mask_losses, er_losses, sep_losses = [], [], [], [], [], [], []

    model.train()
    scaler = amp.GradScaler()
    processor = dist_logger.init_processor(data_loader)
    for img, gc, gbm in processor:
        img, gc, gbm = img.cuda(), gc.cuda(), gbm.cuda()

        with amp.autocast():
            c, im = model(img)
            cla_loss, mask_loss = criterion(c, None, gc, im, None, gbm)
            # loss = criterion.class_weight * (cla_loss + aff_cla_loss) + criterion.mask_weight * (mask_loss + aff_mask_loss) + criterion.er_weight * er_loss  # + sep_loss
            loss = criterion.class_weight * cla_loss + criterion.mask_weight * mask_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.clone().detach())
        cla_losses.append(cla_loss.clone().detach())
        mask_losses.append(mask_loss.clone().detach())
        # aff_cla_losses.append(aff_cla_loss.clone().detach())
        # aff_mask_losses.append(aff_mask_loss.clone().detach())
        # er_losses.append(er_loss.clone().detach())
        # sep_losses.append(sep_loss.clone().detach())

        cur_loss = dist_logger.reduce_tensor(loss)
        avg_loss = dist_logger.reduce_epoch_loss(losses)
        dist_logger.update_processor(processor, f'Epoch: {epoch_idx + 1}, avg_loss: {avg_loss:.2f}, cur_loss: {cur_loss:.2f}')

    lr_scheduler.step()

    dist_logger.save_model(model)
    dist_logger.update_tensorboard(super_tag='loss', tag_scaler_dict={
        'loss': dist_logger.reduce_epoch_loss(losses),
        'cla': dist_logger.reduce_epoch_loss(cla_losses),
        'mask': dist_logger.reduce_epoch_loss(mask_losses),
        # 'aff-cla': dist_logger.reduce_epoch_loss(aff_cla_losses),
        # 'aff-mask': dist_logger.reduce_epoch_loss(aff_mask_losses),
        # 'er': dist_logger.reduce_epoch_loss(er_losses),
        # 'sep': dist_logger.reduce_epoch_loss(sep_losses)
    }, idx=epoch_idx)


@torch.no_grad()
def val_one_epoch(model, data_loader, coco_gt, dist_logger, epoch_idx):
    pred_instances = []

    model.train()
    processor = dist_logger.init_processor(data_loader)
    for img, img_id in processor:
        img = img.cuda()
        img_info_list = coco_gt.loadImgs(img_id.numpy())

        cla_logist, instance_map = model(img)
        instance_map = instance_map.sigmoid()
        cla_conf, cla_pred = torch.max(cla_logist.softmax(dim=-1), dim=-1)

        cla_conf, cla_pred = cla_conf.cpu().numpy(), cla_pred.cpu().numpy()
        batch_size, num_instances = cla_conf.shape

        for batch_idx in range(batch_size):
            for instance_idx in range(num_instances):
                if cla_pred[batch_idx, instance_idx] != 0:
                    org_h = img_info_list[batch_idx]['height']
                    org_w = img_info_list[batch_idx]['width']

                    det_instance_map = instance_map[batch_idx, instance_idx]
                    # det_instance_map -= torch.min(det_instance_map)
                    # det_instance_map /= torch.max(det_instance_map)
                    det_instance_map = func.interpolate(det_instance_map[None, None, ...], size=(org_h, org_w), mode='bilinear', align_corners=True).squeeze()

                    bin_mask = det_instance_map.cpu().numpy() > 0.3
                    bin_mask = np.array(bin_mask[..., None], dtype=np.uint8, order="F")

                    # import cv2
                    # import os
                    # from datasets.tools import CLASSES
                    #
                    # org_img = cv2.imread(os.path.join('../../DataSet/AugVoc2012/val', img_info_list[batch_idx]['file_name']))
                    # mask = det_instance_map.cpu().numpy()[..., None]
                    # color_map = cv2.applyColorMap(np.array(mask * 255., np.uint8), cv2.COLORMAP_JET)
                    # hot_map = org_img * 0.5 + color_map * 0.5
                    # cv2.imwrite(f'./output/one-conv-in-is/heat_map/{img_id[batch_idx]}_{CLASSES[cla_pred[batch_idx, instance_idx]]}.png', np.array(hot_map, dtype=np.uint8))

                    rle = coco_mask.merge(coco_mask.encode(bin_mask))
                    rle['counts'] = rle['counts'].decode('utf-8')
                    pred_instances.append({
                        'image_id': int(img_id[batch_idx]),
                        'category_id': int(cla_pred[batch_idx, instance_idx]),
                        'segmentation': rle,
                        'score': float(str('%.1f' % cla_conf[batch_idx, instance_idx]))
                    })

    dist_logger.save_pred_instances_local_rank(pred_instances)
    dist_logger.save_val_file()
    dist_logger.update_tensorboard_val_results(coco_gt, epoch_idx)
