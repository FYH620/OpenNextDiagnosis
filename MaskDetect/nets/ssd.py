import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from nets.mobilenetv2 import InvertedResidual, mobilenet_v2
from nets.vgg import vgg as add_vgg
from config import fpn, se


class RB(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1):
        super(RB, self).__init__()
        self.rb = nn.Sequential(
            nn.Conv2d(nin, nout, ksize, stride, pad),
            nn.BatchNorm2d(nout),
            nn.ReLU(inplace=True),
            nn.Conv2d(nin, nout, ksize, stride, pad),
            nn.BatchNorm2d(nout),
        )

    def forward(self, input):
        x = input
        x = self.rb(x)
        return F.relu(input + x)


class SE(nn.Module):
    def __init__(self, nin, nout, reduce=16):
        super(SE, self).__init__()
        self.gp = nn.AdaptiveAvgPool2d(1)
        self.rb1 = RB(nin, nout)
        self.se = nn.Sequential(
            nn.Linear(nout, nout // reduce),
            nn.ReLU(inplace=True),
            nn.Linear(nout // reduce, nout),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = input
        x = self.rb1(x)
        b, c, _, _ = x.size()
        y = self.gp(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        out = y + input
        return out


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


def add_extras(in_channels, backbone_name):
    layers = []
    if backbone_name == "vgg":
        layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]
        layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
    else:
        layers += [InvertedResidual(in_channels, 512, stride=2, expand_ratio=0.2)]
        layers += [InvertedResidual(512, 256, stride=2, expand_ratio=0.25)]
        layers += [InvertedResidual(256, 256, stride=2, expand_ratio=0.5)]
        layers += [InvertedResidual(256, 64, stride=2, expand_ratio=0.25)]
    return nn.ModuleList(layers)


class SSD300(nn.Module):
    def __init__(self, num_classes, backbone_name, pretrained=False):
        super(SSD300, self).__init__()
        self.num_classes = num_classes
        if backbone_name == "vgg":
            self.vgg = add_vgg(pretrained)
            self.extras = add_extras(1024, backbone_name)
            self.L2Norm = L2Norm(512, 20)
            mbox = [4, 6, 6, 6, 4, 4]
            loc_layers = []
            conf_layers = []
            backbone_source = [21, -2]
            for k, v in enumerate(backbone_source):
                loc_layers += [
                    nn.Conv2d(
                        self.vgg[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1
                    )
                ]
                conf_layers += [
                    nn.Conv2d(
                        self.vgg[v].out_channels,
                        mbox[k] * num_classes,
                        kernel_size=3,
                        padding=1,
                    )
                ]
            for k, v in enumerate(self.extras[1::2], 2):
                loc_layers += [
                    nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)
                ]
                conf_layers += [
                    nn.Conv2d(
                        v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1
                    )
                ]
        else:
            self.mobilenet = mobilenet_v2(pretrained).features
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]
            loc_layers = []
            conf_layers = []
            backbone_source = [13, -1]
            for k, v in enumerate(backbone_source):
                loc_layers += [
                    nn.Conv2d(
                        self.mobilenet[v].out_channels,
                        mbox[k] * 4,
                        kernel_size=3,
                        padding=1,
                    )
                ]
                conf_layers += [
                    nn.Conv2d(
                        self.mobilenet[v].out_channels,
                        mbox[k] * num_classes,
                        kernel_size=3,
                        padding=1,
                    )
                ]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [
                    nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)
                ]
                conf_layers += [
                    nn.Conv2d(
                        v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1
                    )
                ]
        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)
        self.backbone_name = backbone_name

        if fpn:
            self.latten0 = nn.Conv2d(512, 256, 1)
            self.latten1 = nn.Conv2d(1024, 256, 1)
            self.latten2 = nn.Conv2d(512, 256, 1)
            self.latten3 = nn.Conv2d(256, 512, 3, 1, 1)
            self.latten4 = nn.Conv2d(256, 1024, 3, 1, 1)
            self.latten5 = nn.Conv2d(256, 512, 3, 1, 1)
        if se:
            self.se0 = SE(512, 512)
            self.se1 = SE(1024, 1024)
            self.se2 = SE(512, 512)
            self.se3 = SE(256, 256)
            self.se4 = SE(256, 256)
            self.se5 = SE(256, 256)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()
        if self.backbone_name == "vgg":
            for k in range(23):
                x = self.vgg[k](x)
        else:
            for k in range(14):
                x = self.mobilenet[k](x)
        s = self.L2Norm(x)
        sources.append(s)

        if self.backbone_name == "vgg":
            for k in range(23, len(self.vgg)):
                x = self.vgg[k](x)
        else:
            for k in range(14, len(self.mobilenet)):
                x = self.mobilenet[k](x)

        sources.append(x)
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if self.backbone_name == "vgg":
                if k % 2 == 1:
                    sources.append(x)
            else:
                sources.append(x)

        if fpn:
            o2 = self.latten2(sources[2]) + F.upsample(
                sources[3], sources[2].shape[2:], mode="bilinear"
            )
            sources[2] = self.latten5(o2)
            o1 = self.latten1(sources[1]) + F.upsample(
                o2, sources[1].shape[2:], mode="bilinear"
            )
            sources[1] = self.latten4(o1)
            o0 = self.latten0(sources[0]) + F.upsample(
                o1, sources[0].shape[2:], mode="bilinear"
            )
            sources[0] = self.latten3(o0)
        if se:
            sources[0] = self.se0(sources[0])
            sources[1] = self.se1(sources[1])
            sources[2] = self.se2(sources[2])
            sources[3] = self.se3(sources[3])
            sources[4] = self.se4(sources[4])
            sources[5] = self.se5(sources[5])

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
        )
        return output
