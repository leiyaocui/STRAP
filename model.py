import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from vit import _make_pretrained_vitb_rn50_384, forward_vit


class DPT(nn.Module):
    def __init__(
        self, features=256, channels_last=False, use_bn=False, expand=False,
    ):
        super(DPT, self).__init__()

        self.channels_last = channels_last

        self.pretrained, self.scratch = _make_encoder(
            features, use_pretrained=True, groups=1, expand=expand, hooks=[0, 1, 8, 11],
        )

        if expand == True:
            self.scratch.refinenet01 = _make_fusion_block(
                features, use_bn, expand=False
            )
            self.scratch.refinenet02 = _make_fusion_block(
                features * 2, use_bn, expand=True
            )
            self.scratch.refinenet03 = _make_fusion_block(
                features * 4, use_bn, expand=True
            )
            self.scratch.refinenet04 = _make_fusion_block(
                features * 8, use_bn, expand=True
            )
        else:
            self.scratch.refinenet01 = _make_fusion_block(
                features, use_bn, expand=False
            )
            self.scratch.refinenet02 = _make_fusion_block(
                features, use_bn, expand=False
            )
            self.scratch.refinenet03 = _make_fusion_block(
                features, use_bn, expand=False
            )
            self.scratch.refinenet04 = _make_fusion_block(
                features, use_bn, expand=False
            )


class DPTAffordanceModel(DPT):
    def __init__(self, num_classes, features=256):
        assert num_classes > 0
        super().__init__(
            features=features, use_bn=True, expand=False,
        )

        self.num_classes = num_classes

        self.head_dict = nn.ModuleDict()
        for i in range(self.num_classes):
            self.head_dict[str(i)] = _make_head(features)

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet04(layer_4_rn)
        path_3 = self.scratch.refinenet03(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet02(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet01(path_2, layer_1_rn)

        output = []
        for i in range(self.num_classes):
            head_func = self.head_dict[str(i)]
            output.append(head_func(path_1))

        return output


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super(ResidualConvUnit, self).__init__()

        self.bn = bn

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, padding=1, bias=not self.bn,
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, padding=1, bias=not self.bn,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    def __init__(
        self, features, activation, bn=False, expand=False, align_corners=False
    ):
        super(FeatureFusionBlock, self).__init__()

        self.align_corners = align_corners

        if expand == True:
            out_features = features // 2
        else:
            out_features = features

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, bias=True)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        output = F.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners,
        )

        output = self.out_conv(output)

        return output


def _make_encoder(
    features, use_pretrained=True, groups=1, expand=False, hooks=None,
):
    pretrained = _make_pretrained_vitb_rn50_384(use_pretrained, hooks=hooks)
    scratch = _make_scratch(
        [256, 512, 768, 768], features, groups=groups, expand=expand
    )

    return pretrained, scratch


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8
    else:
        out_shape1 = out_shape
        out_shape2 = out_shape
        out_shape3 = out_shape
        out_shape4 = out_shape

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, padding=1, bias=False, groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, padding=1, bias=False, groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, padding=1, bias=False, groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape4, kernel_size=3, padding=1, bias=False, groups=groups,
    )

    return scratch


def _make_fusion_block(features, use_bn, expand=False):
    return FeatureFusionBlock(
        features, nn.ReLU(False), bn=use_bn, expand=expand, align_corners=True
    )


def _make_head(features):
    head = nn.Sequential(
        nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(features),
        nn.ReLU(True),
        nn.Dropout(0.1, False),
        nn.Conv2d(features, 1, kernel_size=1),
    )
    return head


class PAR(nn.Module):
    def __init__(self, dilations, num_iter):
        super(PAR, self).__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = self.get_kernel()
        self.register_buffer("kernel", kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_kernel(self):
        weight = torch.zeros(8, 1, 3, 3)
        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 2] = 1

        weight[5, 0, 2, 0] = 1
        weight[6, 0, 2, 1] = 1
        weight[7, 0, 2, 2] = 1

        return weight

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode="replicate", value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):
        if imgs.shape[-2:] != masks.shape[-2:]:
            imgs = F.interpolate(
                imgs, size=masks.shape[-2:], mode="bilinear", align_corners=True
            )

        b, c, h, w = imgs.shape
        _imgs = self.get_dilated_neighbors(imgs)
        _pos = self.pos.to(_imgs.device)

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _pos_rep = _pos.repeat(b, 1, 1, h, w)

        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)
        _pos_std = torch.std(_pos_rep, dim=self.dim, keepdim=True)

        aff = -((_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2)
        aff = aff.mean(dim=1, keepdim=True)

        pos_aff = -((_pos_rep / (_pos_std + 1e-8) / self.w1) ** 2)
        # pos_aff = pos_aff.mean(dim=1, keepdim=True)

        aff = F.softmax(aff, dim=2) + self.w2 * F.softmax(pos_aff, dim=2)

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks
