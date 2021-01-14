import argparse
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.parallel as parallel
import torch.optim as optim
import torch.utils.data as data

import engine
import utils.lr_lambda as lr_lambda
from datasets import AugVocTrainDataset, AugVocValDataset, CLASSES
from model import InstanceSetPredictionNet, Criterion
from utils.distributed_logger import DistributedLogger

warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--is_train', default=True, type=bool)

    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--random_seed', default=423, type=int)

    parser.add_argument('--input_size_h', default=320, type=int)
    parser.add_argument('--input_size_w', default=320, type=int)
    parser.add_argument('--num_instances', default=40, type=int)
    parser.add_argument('--d_model', default=128, type=int)

    parser.add_argument('--warm_up_epochs', default=5, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)

    parser.add_argument('--class_weight', default=1., type=float)
    parser.add_argument('--mask_weight', default=1.5, type=float)
    parser.add_argument('--er_weight', default=1., type=float)
    parser.add_argument('--no_instance_coef', default=0.1, type=float)

    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--master_rank', default=0, type=int)

    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def __main__():
    args = get_args_parser()
    dist.init_process_group(backend='nccl')

    set_random_seed(args.random_seed + dist.get_rank())

    torch.cuda.set_device(torch.device('cuda:{}'.format(dist.get_rank())))

    dist_logger = DistributedLogger(args.name, args.master_rank, use_tensorboard=True)

    train_dataset = AugVocTrainDataset(args.dataset_root, args.input_size_w, args.input_size_h, args.num_instances)
    train_sampler = data.distributed.DistributedSampler(train_dataset)
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler, pin_memory=True, drop_last=True)

    val_dataset = AugVocValDataset(args.dataset_root, args.input_size_w, args.input_size_h)
    val_sampler = data.distributed.DistributedSampler(val_dataset)
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)

    model = InstanceSetPredictionNet(num_classes=len(CLASSES), num_instances=args.num_instances, d_model=args.d_model).cuda()
    # model.load_state_dict(torch.load(f'./output/{args.name}/model/param.pth'))

    model = parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()], find_unused_parameters=True)
    criterion = Criterion(args.class_weight, args.mask_weight, args.no_instance_coef, args.er_weight)

    optimizer = optim.SGD(model.module.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda.get_warm_up_cosine_lr_lambda(args.warm_up_epochs, cosine_epochs=args.epochs))

    # engine.val_one_epoch(model, val_dataloader, val_dataset.coco, dist_logger, 0)
    # exit(-1)

    for epoch_idx in range(args.warm_up_epochs + args.epochs):
        train_sampler.set_epoch(epoch_idx)
        engine.train_one_epoch(model, optimizer, criterion, lr_scheduler, train_dataloader, dist_logger, epoch_idx)

        val_sampler.set_epoch(epoch_idx)
        engine.val_one_epoch(model, val_dataloader, val_dataset.coco, dist_logger, epoch_idx)


if __name__ == '__main__':
    # __main__()
    import torch.nn as nn
    from model.instance_set_prediction_transformer import InstanceSetPredictionTransformer

    num_instances = 40
    num_classes = 21
    patch_size = 32
    d_model = 512
    inp = torch.rand((2, 3, 320, 320))

    m = InstanceSetPredictionTransformer(num_instances, num_classes, 100, patch_size, d_model)
    cla_logist, bbox_pred = m(inp)
    print(cla_logist.shape, bbox_pred.shape)
