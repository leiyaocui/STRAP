import torch
from torch import nn

from vit import _make_pretrained_vitb_rn50_384, forward_vit, forward_flex


class Cerberus(nn.Module):
    def __init__(
        self,
        features=256,
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):
        super(Cerberus, self).__init__()

        self.channels_last = channels_last

        self.pretrained, self.scratch = _make_encoder(
            features,
            use_pretrained=True,
            groups=1,
            expand=False,
            hooks=[0, 1, 8, 11],
            use_vit_only=False,
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet01 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet02 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet03 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet04 = _make_fusion_block(features, use_bn)


class CerberusAffordanceModel(Cerberus):
    def __init__(self, num_classes, **kwargs):
        assert num_classes > 0
        features = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True
        super().__init__(**kwargs)

        self.num_classes = num_classes

        self.sigma = nn.Module()
        for i in range(self.num_classes):
            setattr(
                self.sigma,
                f"output_{i}",
                nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(True),
                    nn.Dropout(0.1, False),
                    nn.Conv2d(features, 2, kernel_size=1),
                ),
            )

    def get_attention(self, x, name):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        x = forward_flex(self.pretrained.model, x, True, name)

        return x

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet04(layer_3_rn.shape[-2:], layer_4_rn)
        path_3 = self.scratch.refinenet03(layer_2_rn.shape[-2:], path_4, layer_3_rn)
        path_2 = self.scratch.refinenet02(layer_1_rn.shape[-2:], path_3, layer_2_rn)
        path_1 = self.scratch.refinenet01(None, path_2, layer_1_rn)

        output = []
        for i in range(self.num_classes):
            func = eval(f"self.sigma.output_{i}")
            out = func(path_1)
            out = nn.functional.interpolate(
                out, size=x.shape[-2:], mode="bilinear", align_corners=True
            )
            output.append(out)

        return output


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super(ResidualConvUnit, self).__init__()

        self.bn = bn

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=1,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=1,
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
        self, features, activation, bn=False, expand=False, align_corners=True
    ):
        super(FeatureFusionBlock, self).__init__()

        self.align_corners = align_corners

        if expand == True:
            out_features = features // 2
        else:
            out_features = features

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, next_size, *xs):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if next_size is None:
            output = nn.functional.interpolate(
                output,
                scale_factor=2,
                mode="bilinear",
                align_corners=self.align_corners,
            )
        else:
            output = nn.functional.interpolate(
                output,
                size=next_size,
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
    use_vit_only=False,
    use_readout="ignore",
    enable_attention_hooks=False,
):
    pretrained = _make_pretrained_vitb_rn50_384(
        use_pretrained,
        hooks=hooks,
        use_vit_only=use_vit_only,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
    scratch = _make_scratch(
        [256, 512, 768, 768], features, groups=groups, expand=expand
    )  # ViT-H/16 - 85.0% Top1 (backbone)

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
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock(
        features, nn.ReLU(False), bn=use_bn, expand=False, align_corners=True
    )

