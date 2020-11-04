import torch.nn as nn
import torch.nn.intrinsic as intrinsic
import torch.nn.functional as func
import torch.optim as optim
import torch.utils.data
import torch.utils.tensorboard
import torchvision.models as cv_models
import torchvision.ops as cv_ops
import tqdm
import copy

import datasets.voc as voc


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        backbone_modules = list(cv_models.resnet50(pretrained=True).children())[:-2]
        self.backbone_c2 = nn.Sequential(*backbone_modules[:-3])
        self.backbone_c3 = backbone_modules[-3]
        self.backbone_c4 = backbone_modules[-2]
        self.backbone_c5 = backbone_modules[-1]

        self.neck = cv_ops.FeaturePyramidNetwork(in_channels_list=[256, 512, 1024, 2048], out_channels=256)
        self.output_conv = nn.Sequential(nn.Upsample(scale_factor=4), nn.Conv2d(in_channels=256, out_channels=len(voc.CLASSES), kernel_size=1, bias=False))

    def forward(self, x):
        c2 = self.backbone_c2(x)
        c3 = self.backbone_c3(c2)
        c4 = self.backbone_c4(c3)
        c5 = self.backbone_c5(c4)

        fpn_output_dict = self.neck({'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5})
        p2, p3, p4, p5 = fpn_output_dict.values()

        output = self.output_conv(p2)

        return output


def __main__():
    writer = torch.utils.tensorboard.SummaryWriter(comment='fine-conv-lr-0.01')
    net = torch.nn.DataParallel(Net()).train().cuda()
    dataset = voc.VocSemanticSeg(root='../../DataSet/PASCAL VOC')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=16, shuffle=True, drop_last=True)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60)
    for i in range(150):
        processor = tqdm.tqdm(data_loader)
        losses = []
        for img, seg_mask in processor:
            img = img.cuda()
            seg_mask = seg_mask.cuda()

            optimizer.zero_grad()

            output = net(img)
            loss = nn.functional.cross_entropy(output, seg_mask)
            loss.backward()
            optimizer.step()
            losses.append(loss)
            processor.set_description('Epoch: %d/%d, loss: %.4f, avg_loss: %.4f' % (i + 1, 150, float(loss), sum(losses) / len(losses)))

        lr_scheduler.step()
        writer.add_scalar('avg_loss', sum(losses) / len(losses), i)


if __name__ == '__main__':
    __main__()
