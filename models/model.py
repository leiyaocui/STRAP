from torch import nn
import torch.nn.functional as F

from models.vit import _make_pretrained_vitb_rn50_384, forward_vit
from models.cri_head import CRI


class DPT(nn.Module):
    def __init__(
        self,
        features=256,
        use_bn=False,
        expand=False,
    ):
        super(DPT, self).__init__()

        self.pretrained, self.scratch = _make_encoder(
            features,
            use_pretrained=True,
            groups=1,
            expand=expand,
            hooks=[0, 1, 8, 11],
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
    def __init__(self, num_objects, num_classes, features=256, use_hf=False):
        assert num_classes > 0
        super().__init__(
            features=features,
            use_bn=True,
            expand=False,
        )

        self.num_classes = num_classes

        self.head_dict = nn.ModuleDict()
        for i in range(self.num_classes):
            self.head_dict[str(i)] = _make_head(features)

        if use_hf:
            self.hierarchical_head = nn.Sequential(
                nn.Flatten(1, -1),
                CRI(features * 160 * 160, tuple([num_objects, self.num_classes]))
            )

    def forward(self, x, with_hc=False):
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
            out = self.head_dict[str(i)](path_1)
            out = F.interpolate(out, x.shape[-2:], mode="bilinear", align_corners=False)
            output.append(out)

        if with_hc:
            output_h = self.hierarchical_head(path_1)
            return output, output_h
        else:
            return output


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super(ResidualConvUnit, self).__init__()

        self.bn = bn

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1,
            bias=not self.bn,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1,
            bias=not self.bn,
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
            output,
            scale_factor=2,
            mode="bilinear",
            align_corners=self.align_corners,
        )

        output = self.out_conv(output)

        return output


def _make_encoder(
    features,
    use_pretrained=True,
    groups=1,
    expand=False,
    hooks=None,
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
        in_shape[0],
        out_shape1,
        kernel_size=3,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch


def _make_fusion_block(features, use_bn, expand=False):
    return FeatureFusionBlock(
        features, nn.ReLU(False), bn=use_bn, expand=expand, align_corners=True
    )


def _make_head(features):
    head = nn.Module()

    head = nn.Sequential(
        nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(features),
        nn.ReLU(True),
        nn.Dropout(0.1, False),
        nn.Conv2d(features, 1, kernel_size=1),
    )

    return head
