import torch
import torch.nn.functional as F


def gated_crf_loss(x, y_hat, kernels_desc, kernels_radius, valid_mask=None):
    N, C, H, W = y_hat.shape
    device = y_hat.device

    kernels_diameter = 2 * kernels_radius + 1

    if C == 1:
        y_hat = y_hat.sigmoid()
        y_hat = torch.cat([1 - y_hat, y_hat], dim=1)
    else:
        y_hat = y_hat.softmax(dim=1)

    kernels = 0
    for desc in kernels_desc:
        weight = desc["weight"]
        features = []
        for modality, sigma in desc.items():
            if modality == "weight":
                continue
            elif modality == "xy":
                feature = torch.cat(
                    (
                        torch.arange(
                            0, W, 1, dtype=torch.float32, device=device).view(
                                1, 1, 1, W).repeat(N, 1, H, 1),
                        torch.arange(
                            0, H, 1, dtype=torch.float32, device=device).view(
                                1, 1, H, 1).repeat(N, 1, 1, W),
                    ),
                    1,
                )
            else:
                feature = x

            feature /= sigma
            features.append(feature)

        features = torch.cat(features, dim=1)

        n, c, h, w = features.shape
        kernel = F.unfold(features, kernels_diameter, 1,
                          kernels_radius).view(n, c, kernels_diameter,
                                               kernels_diameter, h, w)
        kernel = kernel - kernel[:, :, kernels_radius,
                                 kernels_radius, :, :].view(n, c, 1, 1, h, w)
        kernel = (-0.5 * kernel**2).sum(dim=1, keepdim=True).exp()
        kernel[:, :, kernels_radius, kernels_radius, :, :] = 0

        kernels += weight * kernel

    denom = N * H * W

    if valid_mask is not None:
        denom = valid_mask.sum().clamp(min=1)
        n, c, h, w = valid_mask.shape
        valid_mask = F.unfold(valid_mask, kernels_diameter, 1,
                              kernels_radius).view(n, c, kernels_diameter,
                                                   kernels_diameter, h, w)
        kernels *= valid_mask

    y_hat_unfolded = F.unfold(y_hat, kernels_diameter, 1,
                              kernels_radius).view(N, y_hat.shape[1],
                                                   kernels_diameter,
                                                   kernels_diameter, H, W)

    product_kernels_y_hat = ((kernels * y_hat_unfolded).view(
        N, y_hat.shape[1], kernels_diameter**2, H, W).sum(dim=2,
                                                          keepdim=False))

    loss = -(product_kernels_y_hat * y_hat).sum()
    loss = kernels.sum() + loss
    loss /= denom

    return loss


def bce_loss(x, y, ignore_index=255):
    x = x.flatten()
    y = y.flatten()
    mask = (y != ignore_index)
    x = x[mask]
    y = y[mask]

    z = (x >= 0).float()

    pos_loss = y * torch.log(1 + torch.exp(x - 2 * x * z)) + y * x * (z - 1)
    neg_loss = (1 - y) * torch.log(1 + torch.exp(x - 2 * x * z)) + (1 -
                                                                    y) * x * z

    num_pos = (y == 1).sum().clamp(min=1)
    num_neg = (y == 0).sum().clamp(min=1)

    loss = pos_loss.sum() / num_pos + neg_loss.sum() / num_neg

    return loss


def ce_loss(x, y, ignore_index=255):
    pos_w = (y == 0).sum().clamp(min=1)
    neg_w = (y == 1).sum().clamp(min=1)
    weight = torch.tensor([neg_w, pos_w], dtype=torch.float32, device="cuda")

    loss = F.cross_entropy(x,
                           y,
                           weight=weight,
                           ignore_index=ignore_index,
                           reduction="sum")

    return loss
