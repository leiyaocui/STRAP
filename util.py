import numpy as np
import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.count = 0.0

    def update(self, value, count):
        self.value += value * count
        self.count += count

    def get(self):
        return self.value / self.count


np.seterr(invalid="ignore")


@torch.no_grad()
def IoU(output, target, num_class, ignore_index=255):
    pred = output.int().flatten().cpu().numpy()
    target = target.int().flatten().cpu().numpy()
    mask = target != ignore_index

    pred = pred[mask]
    target = target[mask]

    if target.max() == 0:
        return np.nan
    else:
        hist = np.bincount(num_class * target + pred, minlength=num_class**2).reshape(
            num_class, num_class
        )

        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

        return iou[1] * 100
