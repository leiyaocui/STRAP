import torch
import torch.nn.functional as F


class GatedCRFLoss(torch.nn.Module):
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

        y_hat = y_hat.softmax(dim=1)

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


class SigmoidCrossEntropyLoss(torch.nn.Module):
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


if __name__ == "__main__":
    from time import time

    x = [torch.randn(4, 1, 25, 25) for _ in range(6)]
    y = [(torch.randn(4, 25, 25) > 0.5).long() for _ in range(6)]

    loss_func = SigmoidCrossEntropyLoss(ignore_index=255)
    begin = time()
    loss = []
    for i in range(6):
        loss.append(loss_func(x[i].squeeze(1), y[i], 2))
    loss = sum(loss)
    end = time()
    print(f"loss: {loss.item()}")
    print(f"time(ms): {(end - begin) * 1000}")
