import torch


def convert_bboxes_xywh_xyxy(bboxes):
    cx, cy, w, h = bboxes.unbind(dim=-1)
    b = [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]
    return torch.stack(b, dim=-1)


def get_area(bboxes):
    wh = (bboxes[..., 2:] - bboxes[..., :2]).clamp(min=0.)
    area = wh[..., 0] * wh[..., 1]
    return area


def get_pair_iou(bboxes1, bboxes2):
    # [B, num_instances]
    area1 = get_area(bboxes1)
    area2 = get_area(bboxes2)

    # [B, num_instances, 2]
    lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # left-top coordination of inter-section
    rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # right-bottom coordination of inter-section

    wh = (rb - lt).clamp(min=0.)  # width and height of inter-section
    inter = wh[..., 0] * wh[..., 1]
    union = area1 + area2 - inter

    return inter / (union + 1e-7), union


def get_pair_giou(bboxes1, bboxes2):
    iou, union = get_pair_iou(bboxes1, bboxes2)

    # [B, num_instances, 2]
    lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])  # left-top coordination of closure-section
    rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])  # right-bottom coordination of closure-section

    wh = (rb - lt).clamp(min=0.)
    closure = wh[..., 0] * wh[..., 1]

    return iou - (closure - union) / (union + 1e-7)


def get_mutual_iou(bboxes1, bboxes2):
    # [B, num_instances, num_instances]
    num_instances = bboxes1.shape[1]
    area1 = get_area(bboxes1)[:, :, None].repeat(1, 1, num_instances)
    area2 = get_area(bboxes2)[:, None, :].repeat(1, num_instances, 1)

    # [B, num_instances, num_instances, 2]
    lt = torch.max(bboxes1[:, :, None, :2].repeat(1, 1, num_instances, 1), bboxes2[:, None, :, :2].repeat(1, num_instances, 1, 1))
    rb = torch.min(bboxes1[:, :, None, 2:].repeat(1, 1, num_instances, 1), bboxes2[:, None, :, 2:].repeat(1, num_instances, 1, 1))

    wh = (rb - lt).clamp(min=0.)
    inter = wh[..., 0] * wh[..., 1]
    union = area1 + area2 - inter

    return inter / (union + 1e-7), union


def get_mutual_giou(bboxes1, bboxes2):
    iou, union = get_mutual_iou(bboxes1, bboxes2)

    # [B, num_instances, num_instances, 2]
    num_instances = bboxes1.shape[1]
    lt = torch.min(bboxes1[:, :, None, :2].repeat(1, 1, num_instances, 1), bboxes2[:, None, :, :2].repeat(1, num_instances, 1, 1))
    rb = torch.max(bboxes1[:, :, None, 2:].repeat(1, 1, num_instances, 1), bboxes2[:, None, :, 2:].repeat(1, num_instances, 1, 1))

    wh = (rb - lt).clamp(min=0.)
    closure = wh[..., 0] * wh[..., 1]

    return iou - (closure - union) / (union + 1e-7)


def recover_bboxes(bboxes, ow, oh):
    bboxes[:, [0, 2]] *= ow
    bboxes[:, [1, 3]] *= oh
    bboxes[:, :2] -= bboxes[:, 2:] / 2
    return bboxes
