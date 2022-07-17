import os
from PIL import Image
import numpy as np


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def IoU(output, target):
    num_classes = output.shape[1]

    pred = output.argmax(dim=1).cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    
    mask = (target >= 0) & (target < num_classes)

    hist = np.bincount(
        num_classes * target[mask] + pred[mask], minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    ious = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    
    return ious * 100


def save_colorful_image(data, file_name, save_dir, palettes):
    save_path = os.path.join(save_dir, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = data.cpu().numpy().squeeze(0)
    img = Image.fromarray(palettes[data])
    img.save(save_path)
