import torch
from torch import nn
from torch.nn import functional as F

import layer.extractors as extractors

# This class if from https://github.com/j96w/DenseFusion
class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(features, size) for size in sizes],
        )
        self.bottleneck = nn.Conv2d(
            features * (len(sizes) + 1), out_features, kernel_size=1,
        )
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.upsample(
                input=stage(feats), size=(
                    h, w,
                ), mode='bilinear',
            ) for stage in self.stages
        ] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

# This class if from https://github.com/j96w/DenseFusion
class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.conv(x)

# This class if from https://github.com/j96w/DenseFusion
class PSPNet(nn.Module):
    def __init__(
        self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet18',
        pretrained=False,
    ):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
            # nn.LogSoftmax()
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)

        return self.final(p)


psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152'),
}


class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super().__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = ModifiedResnet()

        self.f_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.f_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.f_conv3 = torch.nn.Conv1d(128, 256, 1)
        self.f_conv4 = torch.nn.Conv1d(256, 512, 1)
        self.f_conv5 = torch.nn.Conv1d(512, 1024, 1)

        self.c_conv1 = torch.nn.Conv1d(1024, 512, 1)
        self.c_conv2 = torch.nn.Conv1d(512, 256, 1)

        self.c_conv3 = torch.nn.Conv1d(256, 128, 1)
        self.c_conv4 = torch.nn.Conv1d(128, 64, 1)
        self.c_conv5 = torch.nn.Conv1d(64, 1, 1)

        self.conv1_r = torch.nn.Conv1d(1024, 512, 1)
        self.conv1_t = torch.nn.Conv1d(1024, 512, 1)

        self.conv2_r = torch.nn.Conv1d(512, 256, 1)
        self.conv2_t = torch.nn.Conv1d(512, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, 3, 1)  # quaternion
        self.conv4_t = torch.nn.Conv1d(128, 1, 1)  # translation

    def forward(self, img):
        out_img = self.cnn(img)

        bs, di, _, _ = out_img.size()
        emb = out_img.view(bs, di, -1)
        emb = F.relu(self.f_conv1(emb))
        emb = F.relu(self.f_conv2(emb))
        emb = F.relu(self.f_conv3(emb))
        emb = F.relu(self.f_conv4(emb))
        emb = F.relu(self.f_conv5(emb))
        conf = F.relu(self.c_conv1(emb))
        conf = F.relu(self.c_conv2(conf))
        conf = F.relu(self.c_conv3(conf))
        conf = F.relu(self.c_conv4(conf))
        conf = F.relu(self.c_conv5(conf))
        conf = conf.repeat(1, 1024, 1)
        conf = F.softmax(conf, dim=2)

        emb = conf*emb
        _, _, pixels = emb.size()
        ap_x = F.avg_pool1d(emb, kernel_size=pixels)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))

        rx = self.conv4_r(rx)
        tx = self.conv4_t(tx)
        return rx, tx
