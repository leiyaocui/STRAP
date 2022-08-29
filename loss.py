import torch
import torch.nn.functional as F
from kernels.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
from kernels.lib_tree_filter.modules.tree_filter import TreeFilter2D


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
    pos_w = torch.clamp(1 / (y == 1).sum(), max=1.0)
    neg_w = torch.clamp(1 / (y == 0).sum(), max=1.0)
    weight = torch.tensor([neg_w, pos_w], dtype=torch.float32, device="cuda")

    loss = F.cross_entropy(x,
                           y,
                           weight=weight,
                           ignore_index=ignore_index,
                           reduction="sum")

    return loss


def aff_loss(x, y, ignore_index=255):
    x = x.flatten()
    y = y.flatten()
    mask = (y != ignore_index)
    x = x[mask]
    y = y[mask]

    loss = torch.mean(y * (1 - x)) + torch.mean((1 - y) * x)
    loss *= 0.5

    return loss


class SparseCRFLoss(torch.nn.Module):
    def __init__(self, sigma=0.15, diag=False):
        super(SparseCRFLoss, self).__init__()
        self.sigma = sigma
        self.diag = diag

    def forward(self, img, y_pr):
        # regularized loss is not applied to the background class
        assert y_pr.dim == 4 and y_pr.shape[1] == 1

        mask_h, mask_v = self.compute_edge_mask(img)

        loss = self.regularized_loss_per_channel(mask_h, mask_v, 0, y_pr)

        if self.diag:
            mask_d1, mask_d2 = self.compute_edge_mask(img, diag=True)
            loss = loss + self.regularized_loss_per_channel(
                mask_d1, mask_d2, 0, y_pr, diag=True)

        return loss

    def compute_edge_mask(self, image, diag=False):
        if diag:
            if image.shape[1] > 1:
                image = torch.sum(image, dim=1)
            left_ = image[:, :-1, :]
            left_ = left_[:, :, 1:]
            diag1 = image[:, 1:, :]
            diag1 = diag1[:, :, :-1]

            top_ = image[:, :, :-1]
            top_ = top_[:, :-1, :]
            diag2 = image[:, :, 1:]
            diag2 = diag2[:, 1:, :]

            mask_h = torch.exp(-1 * (left_ - diag1)**2 /
                               (2 * self.sigma**2)) * 0.707
            mask_v = torch.exp(-1 * (top_ - diag2)**2 /
                               (2 * self.sigma**2)) * 0.707
        else:
            left_0 = image[:, 0, :-1, :]
            right_0 = image[:, 0, 1:, :]
            top_0 = image[:, 0, :, :-1]
            bottom_0 = image[:, 0, :, 1:]

            left_1 = image[:, 1, :-1, :]
            right_1 = image[:, 1, 1:, :]
            top_1 = image[:, 1, :, :-1]
            bottom_1 = image[:, 1, :, 1:]

            left_2 = image[:, 2, :-1, :]
            right_2 = image[:, 2, 1:, :]
            top_2 = image[:, 2, :, :-1]
            bottom_2 = image[:, 2, :, 1:]

            mask_h = torch.exp(-1 *
                               ((left_0 - right_0)**2 + (left_1 - right_1)**2 +
                                (left_2 - right_2)**2) / (2 * self.sigma**2))
            mask_v = torch.exp(-1 *
                               ((top_0 - bottom_0)**2 + (top_1 - bottom_1)**2 +
                                (top_2 - bottom_2)**2) / (2 * self.sigma**2))

        return mask_h, mask_v

    def regularized_loss_per_channel(self,
                                     mask_h,
                                     mask_v,
                                     cl,
                                     prediction,
                                     diag=False):
        if diag:
            left = prediction[:, cl, :-1, :]
            left = left[:, :, 1:]
            diag1 = prediction[:, cl, 1:, :]
            diag1 = diag1[:, :, :-1]

            top = prediction[:, cl, :, :-1]
            top = top[:, :-1, :]
            diag2 = prediction[:, cl, :, 1:]
            diag2 = diag2[:, 1:, :]

            h = torch.mean(abs(left - diag1) * mask_h)
            v = torch.mean(abs(top - diag2) * mask_v)
        else:
            left = prediction[:, cl, :-1, :]
            right = prediction[:, cl, 1:, :]
            top = prediction[:, cl, :, :-1]
            bottom = prediction[:, cl, :, 1:]

            h = torch.mean(abs(left - right) * mask_h)
            v = torch.mean(abs(top - bottom) * mask_v)

        return (h + v) / 2.0


class TreeEnergyLoss(torch.nn.Module):
    def __init__(self, sigma=0.02):
        super(TreeEnergyLoss, self).__init__()
        self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter_layers = TreeFilter2D(groups=1, sigma=sigma)

    def forward(self, preds, low_feats, high_feats, unlabeled_ROIs):
        with torch.no_grad():
            h, w = preds.shape[-2:]
            # low_feats = F.interpolate(low_feats, size=(h, w), mode='bilinear', align_corners=True)
            unlabeled_ROIs = F.interpolate(unlabeled_ROIs,
                                           size=(h, w),
                                           mode='nearest')
            # N = unlabeled_ROIs.sum()
        n = unlabeled_ROIs.sum().item()

        # prob = torch.softmax(preds, dim=1)
        prob = torch.sigmoid(preds)

        # low-level MST
        tree = self.mst_layers(low_feats)
        AS = self.tree_filter_layers(feature_in=prob,
                                     embed_in=low_feats,
                                     tree=tree)  # [b, n, h, w]

        # high-level MST
        tree = self.mst_layers(high_feats)
        AS = self.tree_filter_layers(feature_in=AS,
                                     embed_in=high_feats,
                                     tree=tree,
                                     low_tree=False)  # [b, n, h, w]

        tree_loss = (unlabeled_ROIs * torch.abs(prob - AS)).sum()

        if n > 0:
            tree_loss /= n
        else:
            tree_loss = 0

        return tree_loss


if __name__ == "__main__":
    x = torch.randn((320, 320)).float()
    y = (torch.randn((320, 320)) > 0.5).int()
    # y[(y == 1) | (y == 0)] = 255

    from time import time

    begin_time = time()
    loss = bce_loss(x, y)
    end_time = time()
    print(f"loss: {loss.item()}")
    print(f"time: {end_time - begin_time}")