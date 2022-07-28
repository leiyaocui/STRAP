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


class SigmoidCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(SigmoidCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, x, y):
        x = x.flatten()
        y = y.flatten()
        mask = y != self.ignore_index
        x = x[mask]
        y = y[mask]

        z = (x >= 0).float()

        loss = torch.log(1 + torch.exp(x - 2 * x * z)) + (z - y) * x
        loss = torch.mean(loss)

        return loss


class CyclicalFocalLoss(nn.Module):
    def __init__(
        self,
        gamma_pos=2,
        gamma_neg=2,
        gamma_hc=3,
        eps: float = 0.1,
        epochs=100,
        factor=4,
        ignore_index=255,
    ):
        super(CyclicalFocalLoss, self).__init__()

        self.eps = eps
        self.gamma_hc = gamma_hc
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.epochs = epochs
        self.factor = factor
        self.ignore_index = ignore_index

    def forward(self, inputs, target, epoch):
        assert target.dim() == 3 and target.shape[1:] == inputs.shape[2:]
        num_classes = inputs.shape[1]
        assert num_classes == 2

        mask = (target != self.ignore_index)

        preds = inputs.softmax(1)
        targets = torch.stack([(target != 1), target], dim=1)

        # Cyclical
        if self.factor * epoch < self.epochs:
            eta = 1 - self.factor * epoch / (self.epochs - 1)
        else:
            eta = (self.factor * epoch / (self.epochs - 1) - 1.0) / (self.factor - 1.0)

        # ASL weights
        anti_targets = 1 - targets
        xs_pos = preds
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg,
            self.gamma_pos * targets + self.gamma_neg * anti_targets,
        )
        positive_w = torch.pow(1 + xs_pos, self.gamma_hc * targets)
        log_preds = torch.log(preds) * ((1 - eta) * asymmetric_w + eta * positive_w)

        if self.eps > 0:  # label smoothing
            targets = targets.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = -targets.mul(log_preds)
        loss = loss.sum(dim=1)
        loss = loss[mask]

        return loss.mean()


if __name__ == "__main__":
    x = [torch.randn(4, 2, 25, 25) for _ in range(6)]
    y = [(torch.randn(4, 25, 25) > 0.5).long() for _ in range(6)]

    from time import time

    loss_func = CyclicalFocalLoss()
    begin = time()
    loss = []
    for i in range(6):
        loss.append(loss_func(x[i], y[i], 1))
    loss = sum(loss)
    end = time()
    print(f"loss: {loss.item()}")
    print(f"time(ms): {(end - begin) * 1000}")

    # loss_func = GatedCRFLoss(
    #     kernels_desc=[{"weight": 1, "xy": 6, "image": 0.1}], kernels_radius=5,
    # )
    # image = torch.randn(4, 3, 25, 25)
    # begin = time()
    # loss = []
    # for i in range(6):
    #     loss.append(loss_func(image, x[i]))
    # loss = sum(loss)
    # end = time()
    # print(f"loss: {loss.item()}")
    # print(f"time(ms): {(end - begin) * 1000}")

