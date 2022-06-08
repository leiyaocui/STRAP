import numpy as np


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))


def mIoU(output, target):
    num_classes = output.shape[1]
    pred = output.max(dim=1)[1]
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    hist = fast_hist(pred.flatten(), target.flatten(), num_classes) 
    ious = per_class_iu(hist) * 100
    return ious