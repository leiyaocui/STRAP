import numpy as np
import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


np.seterr(invalid="ignore")


@torch.no_grad()
def IoU(output, target, num_class, ignore_index=255):
    pred = output.int().flatten().cpu().numpy()
    target = target.int().flatten().cpu().numpy()
    mask = target != ignore_index

    pred = pred[mask]
    target = target[mask]

    hist = np.bincount(num_class * target + pred,
                       minlength=num_class**2).reshape(num_class, num_class)
    ious = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    return ious[1] * 100