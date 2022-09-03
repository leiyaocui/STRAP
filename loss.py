import torch
import torch.nn.functional as F


def gated_crf_loss(x, y, kernels_desc, kernels_radius, mask_src=None, mask_dst=None):
    """
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
    :param kernels_radius: Defines size of bounding box region around each pixel in which the kernel is constructed.
    """
    N, C, H, W = y.shape
    device = y.device

    kernels_diameter = 2 * kernels_radius + 1

    y_hat = y.sigmoid()
    y_hat = torch.cat([1 - y_hat, y_hat], dim=1)
    C = 2

    kernels = 0.0
    for desc in kernels_desc:
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
                    dim=1,
                )
            elif "image":
                feature = x.clone()

            feature /= sigma
            features.append(feature)

        features = torch.cat(features, dim=1)

        n, c, h, w = features.shape
        kernel = F.unfold(features, kernels_diameter, padding=kernels_radius).view(
            n, c, kernels_diameter, kernels_diameter, h, w
        )
        kernel = kernel - kernel[:, :, kernels_radius, kernels_radius, :, :].view(
            n, c, 1, 1, h, w
        )
        kernel = (-0.5 * kernel**2).sum(dim=1, keepdim=True).exp()
        kernel[:, :, kernels_radius, kernels_radius, :, :] = 0

        kernels += weight * kernel

    denom = N * H * W

    if mask_src is not None:
        denom = min(mask_src.sum().clamp(min=1), denom)
        n, c, h, w = mask_src.shape
        mask_src_unfolded = F.unfold(
            mask_src, kernels_diameter, padding=kernels_radius
        ).view(n, c, kernels_diameter, kernels_diameter, h, w)
        kernels *= mask_src_unfolded

    if mask_dst is not None:
        denom = min(mask_dst.sum().clamp(min=1), denom)
        mask_dst_unfolded = mask_dst.view(N, 1, 1, 1, H, W)
        kernels *= mask_dst_unfolded

    y_hat_unfolded = F.unfold(y_hat, kernels_diameter, padding=kernels_radius).view(
        N, C, kernels_diameter, kernels_diameter, H, W
    )

    product_kernels_y_hat = (
        (kernels * y_hat_unfolded).view(N, C, kernels_diameter**2, H, W).sum(dim=2)
    )

    loss = kernels.sum() - (product_kernels_y_hat * y_hat).sum()
    loss /= denom

    return loss


def bce_loss(x, y, ignore_index=255):
    x = x.flatten()
    y = y.flatten()
    mask = y != ignore_index
    x = x[mask]
    y = y[mask]

    z = (x >= 0).float()

    pos_loss = y * torch.log(1 + torch.exp(x - 2 * x * z)) + y * x * (z - 1)
    neg_loss = (1 - y) * torch.log(1 + torch.exp(x - 2 * x * z)) + (1 - y) * x * z

    num_pos = (y == 1).sum().clamp(min=1)
    num_neg = (y == 0).sum().clamp(min=1)

    loss = pos_loss.sum() / num_pos + neg_loss.sum() / num_neg

    return loss


def ce_loss(x, y, ignore_index=255):
    pos_w = (y == 0).sum().clamp(min=1)
    neg_w = (y == 1).sum().clamp(min=1)
    weight = torch.tensor([neg_w, pos_w], dtype=torch.float32, device="cuda")

    loss = F.cross_entropy(
        x, y, weight=weight, ignore_index=ignore_index, reduction="sum"
    )

    return loss
