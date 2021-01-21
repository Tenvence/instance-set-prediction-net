import torch
import torch.nn.utils
import torch.cuda.amp as amp

import utils.bbox_ops as bbox_ops


def train_one_epoch(model, optimizer, matcher, criterion, lr_scheduler, data_loader, dist_logger, epoch_idx):
    losses, cla_losses, giou_losses, l1_losses = [], [], [], []

    model.train()
    scaler = amp.GradScaler()
    processor = dist_logger.init_processor(data_loader)
    for img, gc, gb in processor:
        img, gc, gb = img.cuda(), gc.cuda(), gb.cuda()

        with amp.autocast():
            c, b = model(img)
            mgc, mgb = matcher(c, b, gc, gb)
            cla_loss, giou_loss, l1_loss = criterion(c, b, mgc, mgb)
            loss = matcher.class_weight * cla_loss + matcher.giou_weight * giou_loss + matcher.l1_weight * l1_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.clone().detach())
        cla_losses.append(cla_loss.clone().detach())
        giou_losses.append(giou_loss.clone().detach())
        l1_losses.append(l1_loss.clone().detach())

        cur_loss = dist_logger.reduce_tensor(loss)
        avg_loss = dist_logger.reduce_epoch_loss(losses)
        dist_logger.update_processor(processor, f'Epoch: {epoch_idx + 1}, avg_loss: {avg_loss:.2f}, cur_loss: {cur_loss:.2f}')

    lr_scheduler.step()

    dist_logger.save_model(model)
    dist_logger.update_tensorboard(super_tag='loss', tag_scaler_dict={
        'loss': dist_logger.reduce_epoch_loss(losses),
        'cla': dist_logger.reduce_epoch_loss(cla_losses),
        'giou': dist_logger.reduce_epoch_loss(giou_losses),
        'l1': dist_logger.reduce_epoch_loss(l1_losses)
    }, idx=epoch_idx)


@torch.no_grad()
def val_one_epoch(model, data_loader, coco_gt, dist_logger, epoch_idx):
    pred_instances = []

    model.eval()
    processor = dist_logger.init_processor(data_loader)
    for img, img_id in processor:
        img = img.cuda()
        img_info_list = coco_gt.loadImgs(img_id.numpy())

        cla_logist, pred_bbox = model(img)
        cla_conf, cla_pred = torch.max(cla_logist.softmax(dim=-1), dim=-1)  # [B, I]

        cla_conf, cla_pred, pred_bbox = cla_conf.cpu().numpy(), cla_pred.cpu().numpy(), pred_bbox.cpu().numpy()
        batch_size, num_instances = cla_conf.shape

        for batch_idx in range(batch_size):
            recovered_bboxes = bbox_ops.denormalize_bboxes(pred_bbox[batch_idx], h=img_info_list[batch_idx]['height'], w=img_info_list[batch_idx]['width'])
            for instance_idx in range(num_instances):
                if cla_pred[batch_idx, instance_idx] != 0:
                    pred_instances.append({
                        'image_id': int(img_id[batch_idx]),
                        'category_id': int(cla_pred[batch_idx, instance_idx]),
                        'bbox': [float(str('%.1f' % coord)) for coord in recovered_bboxes[instance_idx].tolist()],
                        'score': float(str('%.1f' % cla_conf[batch_idx, instance_idx]))
                    })

    dist_logger.save_pred_instances_local_rank(pred_instances)
    dist_logger.save_val_file()
    dist_logger.update_tensorboard_val_results(coco_gt, epoch_idx)
