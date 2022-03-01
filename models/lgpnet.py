import torch
from torch import nn
import torch.nn.functional as F

import models.resnet as models


def maxp_block(mp_kernel_size, mp_stride, in_channels, out_channels, conv_padding=0):
    """Build maxpooling block"""
    return nn.Sequential(
        nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride, padding=0),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=conv_padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def avgp_block(in_channels, out_channels, bin):
    """Build avgpooling block"""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(bin),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class PoolLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bin):
        super(PoolLayer, self).__init__()
        self.avgp = avgp_block(in_channels, out_channels, bin)
        self.maxp1 = maxp_block(3, 2, in_channels, 1024, 1)
        self.conv = nn.Conv2d(1024, 512, kernel_size=3, padding=0)
        self.bcn = nn.BatchNorm2d(512)
        self.maxp2 = maxp_block(2, 1, 512, out_channels)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        x_avgp = self.avgp(x)
        out.append(F.interpolate(x_avgp, x_size[2:], mode='bilinear', align_corners=True))
        x_maxp = self.maxp1(x)
        x_maxp = self.bcn(F.relu(self.conv(x_maxp)))
        out.append(F.interpolate(self.maxp2(x_maxp), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PoolLayerStu(nn.Module):
    def __init__(self, in_channels, out_channels, bin):
        super(PoolLayerStu, self).__init__()
        self.avgp = avgp_block(in_channels, out_channels, bin)
        self.maxp1 = maxp_block(3, 2, in_channels, 512, 1)
        self.maxp2 = maxp_block(2, 1, 512, out_channels)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        x_avgp = self.avgp(x)
        out.append(F.interpolate(x_avgp, x_size[2:], mode='bilinear', align_corners=True))
        x_maxp = self.maxp1(x)
        out.append(F.interpolate(self.maxp2(x_maxp), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class LGPNet(nn.Module):
    def __init__(self, layers=18, dropout=0.1, classes=32, zoom_factor=8, pretrained=True, bin=6, is_stu=False):
        super(LGPNet, self).__init__()
        assert layers in [18, 50, 101, 152]
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor

        if layers == 18:
            resnet = models.resnet18(pretrained=pretrained)
        elif layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)

        self.conv0 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # for resnet18
        if layers == 18:
            feats_dim = 512
            for name, module in self.layer3.named_modules():
                if 'conv2' in name:
                    module.dilation, module.padding, module.stride = (2, 2), (2, 2), (1, 1)
                if 'downsample.0' in name:
                    module.stride = (1, 1)
                if 'conv1' in name:
                    module.stride = (1, 1)
            for name, module in self.layer4.named_modules():
                if 'conv2' in name:
                    module.dilation, module.padding, module.stride = (4, 4), (4, 4), (1, 1)
                if 'downsample.0' in name:
                    module.stride = (1, 1)
                if 'conv1' in name:
                    module.stride = (1, 1)

            if self.training:
                self.aux = nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=dropout),
                    nn.Conv2d(128, classes, kernel_size=1)
                )

        # for resnet50, 101
        else:
            feats_dim = 2048
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            
            if self.training:
                self.aux = nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=dropout),
                    nn.Conv2d(256, classes, kernel_size=1)
                )
        if not is_stu:
            self.gplayer = PoolLayer(feats_dim, int(feats_dim / 2), bin)
        else:
            self.gplayer = PoolLayerStu(feats_dim, int(feats_dim / 2), bin)
        
        feats_dim *= 2
    
        self.cls = nn.Sequential(
            nn.Conv2d(feats_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        self.pos_output = nn.Conv2d(classes, 1, kernel_size=2)
        self.cos_output = nn.Conv2d(classes, 1, kernel_size=2)
        self.sin_output = nn.Conv2d(classes, 1, kernel_size=2)
        self.width_output = nn.Conv2d(classes, 1, kernel_size=2)

    def forward(self, x, y=None):
        x_size = x.size()
        assert ((x_size[2] == 300) and (x_size[3] == 300))
        h = 301
        w = 301

        x = self.conv0(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)

        x = self.gplayer(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            aux_pos_output = self.pos_output(aux)
            aux_cos_output = self.cos_output(aux)
            aux_sin_output = self.sin_output(aux)
            aux_width_output = self.width_output(aux)

        main_pos_output = self.pos_output(x)
        main_cos_output = self.cos_output(x)
        main_sin_output = self.sin_output(x)
        main_width_output = self.width_output(x)

        ret_main = [main_pos_output, main_cos_output, main_sin_output, main_width_output]
        if self.training:
            ret_aux = [aux_pos_output, aux_cos_output, aux_sin_output, aux_width_output]
            return ret_aux, ret_main
        else:
            return ret_main

    def compute_loss(self, xc, yc):
        if self.training:
            aux_pred, main_pred = self(xc)
            main_loss = [F.mse_loss(main_pred[i], yc[i]) for i in range(len(yc))]
            aux_loss = [0.7 * F.mse_loss(aux_pred[i], yc[i]) for i in range(len(yc))]
            pos_loss = main_loss[0] + aux_loss[0]; cos_loss = main_loss[1] + aux_loss[1]
            sin_loss = main_loss[2] + aux_loss[2]; width_loss = main_loss[3] + aux_loss[3]
        else:
            main_pred = self(xc)
            main_loss = [F.mse_loss(main_pred[i], yc[i]) for i in range(len(yc))]
            pos_loss = main_loss[0]; cos_loss = main_loss[1]; sin_loss = main_loss[2]; width_loss = main_loss[3]

        return {
            'loss': pos_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'pos_loss': pos_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': main_pred[0],
                'cos': main_pred[1],
                'sin': main_pred[2],
                'width': main_pred[3]
            }
        }
