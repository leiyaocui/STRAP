import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCRFLoss(nn.Module):
    def __init__(self, kernels_desc, kernels_radius):
        """
        GatedCRFLoss
        :param kernels_desc: A list of dictionaries, each describing one Gaussian kernel composition from modalities.
            The final kernel is a weighted sum of individual kernels. Following example is a composition of
            RGBXY and XY kernels:
            kernels_desc: [{
                'weight': 0.9,          # Weight of RGBXY kernel
                'xy': 6,                # Sigma for XY
                'image': 0.1,           # Sigma for RGB Image
            },{
                'weight': 0.1,          # Weight of XY kernel
                'xy': 6,                # Sigma for XY
            }]
        :param kernels_radius: Defines size of bounding bo x region around each pixel in which the kernel is constructed.
        """
        super(GatedCRFLoss, self).__init__()
        self.kernels_desc = kernels_desc
        self.kernels_radius = kernels_radius
        self.kernels_diameter = 2 * self.kernels_radius + 1

    def forward(self, x, y_hat, mask_src=None, mask_dst=None):
        assert x.dim() == 4
        assert x.shape[-2:] == y_hat.shape[-2:]
        if mask_src is not None:
            assert x.shape[-2:] == mask_src.shape[-2:] and mask_src.shape[1] == 1
        if mask_dst is not None:
            assert x.shape[-2:] == mask_dst.shape[-2:] and mask_dst.shape[1] == 1

        N, _, H, W = x.shape
        device = x.device

        kernels = 0
        for desc in self.kernels_desc:
            weight = desc["weight"]
            features = []
            for modality, sigma in desc.items():
                if modality == "weight":
                    continue
                elif modality == "xy":
                    feature = torch.cat(
                        (
                            torch.arange(0, W, 1, dtype=torch.float32, device=device)
                            .view(1, 1, 1, W)
                            .repeat(N, 1, H, 1),
                            torch.arange(0, H, 1, dtype=torch.float32, device=device)
                            .view(1, 1, H, 1)
                            .repeat(N, 1, 1, W),
                        ),
                        1,
                    )
                else:
                    feature = x

                feature /= sigma
                features.append(feature)

            features = torch.cat(features, dim=1)

            n, c, h, w = features.shape
            kernel = F.unfold(
                features, self.kernels_diameter, 1, self.kernels_radius
            ).view(n, c, self.kernels_diameter, self.kernels_diameter, h, w)
            kernel = kernel - kernel[
                :, :, self.kernels_radius, self.kernels_radius, :, :
            ].view(n, c, 1, 1, h, w)
            kernel = (-0.5 * kernel ** 2).sum(dim=1, keepdim=True).exp()
            kernel[:, :, self.kernels_radius, self.kernels_radius, :, :] = 0

            kernels += weight * kernel

        denom = N * H * W

        if mask_src is not None:
            denom = mask_src.sum().clamp(min=1)
            mask_src[mask_src != mask_src] = 0.0
            mask_src[mask_src < 1.0] = 0.0
            n, c, h, w = mask_src.shape
            mask_src = F.unfold(
                mask_src, self.kernels_diameter, 1, self.kernels_radius
            ).view(n, c, self.kernels_diameter, self.kernels_diameter, h, w)
            kernels *= mask_src

        if mask_dst is not None:
            mask_dst[mask_dst != mask_dst] = 0.0
            mask_dst[mask_dst < 1.0] = 0.0
            denom = mask_dst.sum().clamp(min=1)
            mask_dst = mask_dst.view(N, 1, 1, 1, H, W)
            kernels *= mask_dst

        y_hat_unfolded = F.unfold(
            y_hat, self.kernels_diameter, 1, self.kernels_radius
        ).view(N, y_hat.shape[1], self.kernels_diameter, self.kernels_diameter, H, W)

        product_kernels_y_hat = (
            (kernels * y_hat_unfolded)
            .view(N, y_hat.shape[1], self.kernels_diameter ** 2, H, W)
            .sum(dim=2, keepdim=False)
        )

        loss = -(product_kernels_y_hat * y_hat).sum()
        loss = kernels.sum() + loss
        loss /= denom

        return loss


class SigmoidGatedCRFLoss(nn.Module):
    def __init__(self, kernels_desc, kernels_radius):
        """
        GatedCRFLoss
        :param kernels_desc: A list of dictionaries, each describing one Gaussian kernel composition from modalities.
            The final kernel is a weighted sum of individual kernels. Following example is a composition of
            RGBXY and XY kernels:
            kernels_desc: [{
                'weight': 0.9,          # Weight of RGBXY kernel
                'xy': 6,                # Sigma for XY
                'image': 0.1,           # Sigma for RGB Image
            },{
                'weight': 0.1,          # Weight of XY kernel
                'xy': 6,                # Sigma for XY
            }]
        :param kernels_radius: Defines size of bounding bo x region around each pixel in which the kernel is constructed.
        """
        super(SigmoidGatedCRFLoss, self).__init__()
        self.kernels_desc = kernels_desc
        self.kernels_radius = kernels_radius
        self.kernels_diameter = 2 * self.kernels_radius + 1

    def forward(self, x, y, mask_src=None, mask_dst=None):
        assert x.dim() == 4 and y.dim() == 3
        assert x.shape[-2:] == y.shape[-2:]
        if mask_src is not None:
            assert x.shape[-2:] == mask_src.shape[-2:] and mask_src.shape[1] == 1
        if mask_dst is not None:
            assert x.shape[-2:] == mask_dst.shape[-2:] and mask_dst.shape[1] == 1

        N, _, H, W = x.shape
        device = x.device

        y_hat = torch.sigmoid(y)
        y_hat = torch.stack([1 - y_hat, y_hat], dim=1)
  
        kernels = 0
        for desc in self.kernels_desc:
            weight = desc["weight"]
            features = []
            for modality, sigma in desc.items():
                if modality == "weight":
                    continue
                elif modality == "xy":
                    feature = torch.cat(
                        (
                            torch.arange(0, W, 1, dtype=torch.float32, device=device)
                            .view(1, 1, 1, W)
                            .repeat(N, 1, H, 1),
                            torch.arange(0, H, 1, dtype=torch.float32, device=device)
                            .view(1, 1, H, 1)
                            .repeat(N, 1, 1, W),
                        ),
                        1,
                    )
                else:
                    feature = x

                feature /= sigma
                features.append(feature)

            features = torch.cat(features, dim=1)

            n, c, h, w = features.shape
            kernel = F.unfold(
                features, self.kernels_diameter, 1, self.kernels_radius
            ).view(n, c, self.kernels_diameter, self.kernels_diameter, h, w)
            kernel = kernel - kernel[
                :, :, self.kernels_radius, self.kernels_radius, :, :
            ].view(n, c, 1, 1, h, w)
            kernel = (-0.5 * kernel ** 2).sum(dim=1, keepdim=True).exp()
            kernel[:, :, self.kernels_radius, self.kernels_radius, :, :] = 0

            kernels += weight * kernel

        denom = N * H * W

        if mask_src is not None:
            denom = mask_src.sum().clamp(min=1)
            mask_src[mask_src != mask_src] = 0.0
            mask_src[mask_src < 1.0] = 0.0
            n, c, h, w = mask_src.shape
            mask_src = F.unfold(
                mask_src, self.kernels_diameter, 1, self.kernels_radius
            ).view(n, c, self.kernels_diameter, self.kernels_diameter, h, w)
            kernels *= mask_src

        if mask_dst is not None:
            mask_dst[mask_dst != mask_dst] = 0.0
            mask_dst[mask_dst < 1.0] = 0.0
            denom = mask_dst.sum().clamp(min=1)
            mask_dst = mask_dst.view(N, 1, 1, 1, H, W)
            kernels *= mask_dst

        y_hat_unfolded = F.unfold(
            y_hat, self.kernels_diameter, 1, self.kernels_radius
        ).view(N, y_hat.shape[1], self.kernels_diameter, self.kernels_diameter, H, W)

        product_kernels_y_hat = (
            (kernels * y_hat_unfolded)
            .view(N, y_hat.shape[1], self.kernels_diameter ** 2, H, W)
            .sum(dim=2, keepdim=False)
        )

        loss = -(product_kernels_y_hat * y_hat).sum()
        loss = kernels.sum() + loss
        loss /= denom

        return loss


class SigmoidCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255, reduction="mean"):
        super(SigmoidCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        assert self.reduction in ["mean", "sum", "none"]

    def forward(self, x, y):
        mask = y != self.ignore_index
        x = x[mask]
        y = y[mask]

        z = (x >= 0).float()

        loss = torch.log(1 + torch.exp(x - 2 * x * z)) + (z - y) * x

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class CyclicalFocalLoss(nn.Module):
    def __init__(
        self,
        gamma_pos=2,
        gamma_neg=2,
        gamma_hc=0,
        eps=1e-8,
        epochs=100,
        factor=2,
        ignore_index=255,
        reduction="mean",
    ):
        super(CyclicalFocalLoss, self).__init__()

        self.eps = eps
        self.gamma_hc = gamma_hc
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.epochs = epochs
        self.factor = factor  # factor=2 for cyclical, 1 for modified

        self.ignore_index = ignore_index
        self.reduction = reduction
        assert self.reduction in ["mean", "sum", "none"]

    def forward(self, inputs, targets, epoch):
        if self.factor * epoch < self.epochs:
            eta = 1 - self.factor * epoch / (self.epochs - 1)
        else:
            eta = (self.factor * epoch / (self.epochs - 1) - 1.0) / (self.factor - 1.0)

        mask = targets != self.ignore_index

        targets = targets[mask].mul(1 - self.eps).add(self.eps)
        anti_targets = 1 - targets
        xs_pos = torch.sigmoid(inputs.squeeze(1)[mask])
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg,
            self.gamma_pos * targets + self.gamma_neg * anti_targets,
        )

        positive_w = torch.pow(1 + xs_pos, self.gamma_hc * targets)
        log_preds = torch.log(xs_pos) * ((1 - eta) * asymmetric_w + eta * positive_w)

        loss = -targets.mul(log_preds)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        # clip=0.05,
        eps=1e-8,
        ignore_index=255,
        reduction="mean",
    ):
        super(AsymmetricLoss, self).__init__()
        self.eps = eps
        # self.clip = clip

        self.ignore_index = ignore_index
        self.reduction = reduction
        assert self.reduction in ["mean", "sum", "none"]

    def forward(self, x, y, gamma_pos, gamma_neg):
        assert gamma_pos >=0 and gamma_neg >= 0
        mask = y != self.ignore_index
        x = x.squeeze(1)[mask]
        y = y[mask]
        anti_y = 1 - y

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        # if self.clip > 0:
        #     xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        loss = -(
            y * torch.log(xs_pos.clamp(min=self.eps))
            + anti_y * torch.log(xs_neg.clamp(min=self.eps))
        )

        # Asymmetric Focusing
        if gamma_neg > 0 or gamma_pos > 0:
            pt = xs_pos * y + xs_neg * anti_y
            asymmetric_w = torch.pow(
                1 - pt, gamma_pos * y + gamma_neg * anti_y
            )
            loss *= asymmetric_w

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


if __name__ == "__main__":
    x = [torch.randn(4, 1, 25, 25) for _ in range(6)]
    y = [(torch.randn(4, 25, 25) > 0.5).long() for _ in range(6)]

    from time import time

    # loss_func = CyclicalFocalLoss()
    # loss_func = AsymmetricLoss()
    # loss_func = SigmoidGatedCRFLoss(
    #     kernels_desc=[{"weight": 1, "xy": 6, "image": 0.1}], kernels_radius=5,
    # )
    loss_func = SigmoidCrossEntropyLoss(ignore_index=255)
    begin = time()
    loss = []
    for i in range(6):
        loss.append(loss_func(x[i].squeeze(1), y[i], 2))
    loss = sum(loss)
    end = time()
    print(f"loss: {loss.item()}")
    print(f"time(ms): {(end - begin) * 1000}")
